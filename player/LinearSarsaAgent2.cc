#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include "LinearSarsaAgent2.h"
#include "LoggerDraw.h"

// If all is well, there should be no mention of anything keepaway- or soccer-
// related in this file.

extern LoggerDraw LogDraw;

LinearSarsaAgent2::LinearSarsaAgent2( int numFeatures, int numActions, bool bLearn,
				    double widths[],
				    char *loadWeightsFile, char *saveWeightsFile ):
  SMDPAgent( numFeatures, numActions )
{
  bLearning = bLearn;

  for ( int i = 0; i < getNumFeatures(); i++ ) {
    tileWidths[ i ] = widths[ i ];
  }

  if ( bLearning && strlen( saveWeightsFile ) > 0 ) {
    strcpy( weightsFile, saveWeightsFile );
    bSaveWeights = true;
  }
  else {
    bSaveWeights = false;
  }

  alpha = 0.125;
  gamma = 1.0;
  lambda = 0;
  epsilon = 0.01;
  minimumTrace = 0.01;

  epochNum = 0;
  lastAction = -1;

  for (int a = 0; a < getNumActions(); a++ ) {
    numNonzeroTraces[a] = 0;
    for ( int i = 0; i < RL_MEMORY_SIZE; i++ ) {
        weights[ a ][ i ] = 0;
        traces[ a ][ i ] = 0;
    }
  }

  srand( (unsigned int) 0 );
  int tmp[ 2 ];
  float tmpf[ 2 ];
  colTab = new collision_table( RL_MEMORY_SIZE, 1 );

  GetTiles( tmp, 1, 1, tmpf, 0 );  // A dummy call to set the hashing table
  srand( time( NULL ) );

  if ( strlen( loadWeightsFile ) > 0 )
    loadWeights( loadWeightsFile );
}

int LinearSarsaAgent2::startEpisode( double state[] )
{
  epochNum++;

  loadTiles( state );
  for ( int a = 0; a < getNumActions(); a++ ) {
    decayTraces( a, 0 );
    Q[ a ] = computeQ( a );
  }

  lastAction = selectAction();

  char buffer[128];
  sprintf( buffer, "Q[%d] = %.2f", lastAction, Q[lastAction] );
  LogDraw.logText( "Qmax", VecPosition( 25, -30 ),
                   buffer,
                   1, COLOR_BROWN );

  for ( int j = 0; j < numTilings; j++ )
    setTrace( lastAction, tiles[ j ], 1.0 );
  return lastAction;
}

int LinearSarsaAgent2::step( double reward, double state[] )
{
  double delta = reward - Q[ lastAction ];
  loadTiles( state );
  for ( int a = 0; a < getNumActions(); a++ ) {
    Q[ a ] = computeQ( a );
  }

  lastAction = selectAction();

  char buffer[128];
  sprintf( buffer, "Q[%d] = %.2f", lastAction, Q[lastAction] );
  LogDraw.logText( "Qmax", VecPosition( 25, -30 ),
                   buffer,
                   1, COLOR_BROWN );

  if ( !bLearning )
    return lastAction;

  //char buffer[128];
  sprintf( buffer, "reward: %.2f", reward );
  LogDraw.logText( "reward", VecPosition( 25, 30 ),
		   buffer,
		   1, COLOR_NAVY );

  delta += Q[ lastAction ];
  updateWeights( delta );
  Q[ lastAction ] = computeQ( lastAction ); // need to redo because weights changed
  decayTraces( lastAction, gamma * lambda );

  for ( int a = 0; a < getNumActions(); a++ ) {  //clear other than F[a]
    if ( a != lastAction ) {
      for ( int j = 0; j < numTilings; j++ )
        clearTrace( a, tiles[ j ] );
    }
  }
  for ( int j = 0; j < numTilings; j++ )      //replace/set traces F[a]
    setTrace( lastAction, tiles[ j ], 1.0 );

  return lastAction;
}

void LinearSarsaAgent2::endEpisode( double reward )
{
  if ( bLearning && lastAction != -1 ) { /* otherwise we never ran on this episode */
    char buffer[128];
    sprintf( buffer, "reward: %.2f", reward );
    LogDraw.logText( "reward", VecPosition( 25, 30 ),
		     buffer,
		     1, COLOR_NAVY );

    /* finishing up the last episode */
    /* assuming gamma = 1  -- if not,error*/
    if ( gamma != 1.0)
      cerr << "We're assuming gamma's 1" << endl;
    double delta = reward - Q[ lastAction ];
    updateWeights( delta );
  }
  if ( bLearning && bSaveWeights && rand() % 200 == 0 ) {
    saveWeights( weightsFile );
  }
  lastAction = -1;
}

int LinearSarsaAgent2::selectAction()
{
  int action;

  // Epsilon-greedy
  if ( bLearning && drand48() < epsilon ) {     /* explore */
    action = rand() % getNumActions();
  }
  else{
    action = argmaxQ();
  }

  return action;
}

bool LinearSarsaAgent2::loadWeights( char *filename )
{
  cout << "Loading weights from " << filename << endl;
  int file = open( filename, O_RDONLY );
  read( file, (char *) weights, RL_MEMORY_SIZE * sizeof(double) );
  colTab->restore( file );
  close( file );
  cout << "...done" << endl;
  return true;
}

bool LinearSarsaAgent2::saveWeights( char *filename )
{
  int file = open( filename, O_CREAT | O_WRONLY, 0664 );
  write( file, (char *) weights, RL_MEMORY_SIZE * sizeof(double) );
  colTab->save( file );
  close( file );
  return true;
}

// Compute an action value from current F and theta
double LinearSarsaAgent2::computeQ( int a )
{
  double q = 0;
  for ( int j = 0; j < numTilings; j++ ) {
    q += weights[ a ][ tiles[ j ] ];
  }

  return q;
}

// Returns index (action) of largest entry in Q array, breaking ties randomly
int LinearSarsaAgent2::argmaxQ()
{
  int bestAction = 0;
  double bestValue = Q[ bestAction ];
  int numTies = 0;
  for ( int a = bestAction + 1; a < getNumActions(); a++ ) {
    double value = Q[ a ];
    if ( value > bestValue ) {
      bestValue = value;
      bestAction = a;
    }
    else if ( value == bestValue ) {
      numTies++;
      if ( rand() % ( numTies + 1 ) == 0 ) {
	bestValue = value;
	bestAction = a;
      }
    }
  }

  return bestAction;
}

void LinearSarsaAgent2::updateWeights( double delta )
{
  double tmp = delta * alpha / numTilings;
  for (int a = 0; a < getNumActions(); a++) {
    for ( int i = 0; i < numNonzeroTraces[a]; i++ ) {
        int f = nonzeroTraces[ a ][ i ];
        if ( f > RL_MEMORY_SIZE || f < 0 )
            cerr << "f is too big or too small!!" << f << endl;
        weights[ a ][ f ] += tmp * traces[ a ][ f ];
    }
  }
}

void LinearSarsaAgent2::loadTiles( double state[] )
{
  int tilingsPerGroup = 32;  /* num tilings per tiling group */
  numTilings = 0;

  /* One tiling for each state variable */
  for ( int v = 0; v < getNumFeatures(); v++ ) {
      GetTiles1( &(tiles[numTilings]), tilingsPerGroup, colTab, state[v]/tileWidths[v], v);
      numTilings += tilingsPerGroup;
  }
  if ( numTilings > RL_MAX_NUM_TILINGS )
    cerr << "TOO MANY TILINGS! " << numTilings << endl;
}


// Clear any trace for feature f
void LinearSarsaAgent2::clearTrace( int a, int f)
{
  if ( f > RL_MEMORY_SIZE || f < 0 )
    cerr << "ClearTrace: f out of range " << f << endl;
  if ( traces[ a ][ f ] != 0 )
    clearExistentTrace(a, f, nonzeroTracesInverse[ a ][ f ] );
}

// Clear the trace for feature f at location loc in the list of nonzero traces
void LinearSarsaAgent2::clearExistentTrace( int a, int f, int loc )
{
  if ( f > RL_MEMORY_SIZE || f < 0 )
    cerr << "ClearExistentTrace: f out of range " << f << endl;
  traces[ a ][ f ] = 0.0;
  numNonzeroTraces[a]--;
  nonzeroTraces[ a ][ loc ] = nonzeroTraces[ a ][ numNonzeroTraces[a] ];
  nonzeroTracesInverse[ a ][ nonzeroTraces[ a ][ loc ] ] = loc;
}

// Decays all the (nonzero) traces by decay_rate, removing those below minimum_trace
void LinearSarsaAgent2::decayTraces( int a, double decayRate )
{
  int f;
  for ( int loc = numNonzeroTraces[a] - 1; loc >= 0; loc-- ) {
    f = nonzeroTraces[a][ loc ];
    if ( f > RL_MEMORY_SIZE || f < 0 )
      cerr << "DecayTraces: f out of range " << f << endl;
    traces[a][ f ] *= decayRate;
    if ( traces[a][ f ] < minimumTrace )
      clearExistentTrace(a, f, loc );
  }
}

// Set the trace for feature f to the given value, which must be positive
void LinearSarsaAgent2::setTrace( int a, int f, float newTraceValue )
{
  if ( f > RL_MEMORY_SIZE || f < 0 )
    cerr << "SetTraces: f out of range " << f << endl;
  if ( traces[ a ][ f ] >= minimumTrace )
    traces[ a ][ f ] = newTraceValue;         // trace already exists
  else {
    while ( numNonzeroTraces[a] >= RL_MAX_NONZERO_TRACES )
      increaseMinTrace(a); // ensure room for new trace
    traces[ a ][ f ] = newTraceValue;
    nonzeroTraces[ a ][ numNonzeroTraces[ a ] ] = f;
    nonzeroTracesInverse[ a ][ f ] = numNonzeroTraces[ a ];
    numNonzeroTraces[a]++;
  }
}

// Try to make room for more traces by incrementing minimum_trace by 10%,
// culling any traces that fall below the new minimum
void LinearSarsaAgent2::increaseMinTrace(int a)
{
  minimumTrace *= 1.1;
  cerr << "Changing minimum_trace to " << minimumTrace << endl;
  for ( int loc = numNonzeroTraces[a] - 1; loc >= 0; loc-- ) { // necessary to loop downwards
    int f = nonzeroTraces[a][ loc ];
    if ( traces[a][ f ] < minimumTrace )
      clearExistentTrace(a, f, loc );
  }
}

void LinearSarsaAgent2::setParams(int iCutoffEpisodes, int iStopLearningEpisodes)
{
  /* set learning parameters */
}
