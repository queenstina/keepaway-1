#include <stdlib.h>
#include <string.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <time.h>
#include "QLearningAgent.h"
#include "LoggerDraw.h"
#include <iostream>

// If all is well, there should be no mention of anything keepaway- or soccer-
// related in this file.

extern LoggerDraw LogDraw;

QLearningAgent::QLearningAgent( int numFeatures, int numActions, bool bLearn,
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
  epsilon = 0.01;

  epochNum = 0;
  lastAction = -1;

  numNonzeroTraces = 0;
  for ( int i = 0; i < RL_MEMORY_SIZE; i++ ) {
    weights[ i ] = 0;
    traces[ i ] = 0;
    R[i] = 0;
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

int QLearningAgent::startEpisode( double state[] )
{
  epochNum++;
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

  return lastAction;
}

int QLearningAgent::step( double reward, double state[] )
{
//  cout << "State:" << endl;
//  for (int i = 0; i < getNumFeatures(); i++) {
//    cout << state[i] << endl;
//  }
//  cout << "----------------" << endl;

  for ( int j = 0; j < numTilings; j++ ) {
      cout << "Reward: " << reward << " .Expected: " << R[tiles[ lastAction ][ j ]] << endl;
  }

  for ( int j = 0; j < numTilings; j++ ) {
    R[tiles[ lastAction ][ j ]] += 0.05*(reward - R[tiles[ lastAction ][ j ]]);
  }

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

  return lastAction;
}

void QLearningAgent::endEpisode( double reward )
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

int QLearningAgent::selectAction()
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

bool QLearningAgent::loadWeights( char *filename )
{
  cout << "Loading weights from " << filename << endl;
  int file = open( filename, O_RDONLY );
  read( file, (char *) weights, RL_MEMORY_SIZE * sizeof(double) );
  colTab->restore( file );
  close( file );
  cout << "...done" << endl;
  return true;
}

bool QLearningAgent::saveWeights( char *filename )
{
  int file = open( filename, O_CREAT | O_WRONLY, 0664 );
  write( file, (char *) weights, RL_MEMORY_SIZE * sizeof(double) );
  colTab->save( file );
  close( file );
  return true;
}

// Compute an action value from current F and theta
double QLearningAgent::computeQ( int a )
{
  double q = 0;
  for ( int j = 0; j < numTilings; j++ ) {
    q += weights[ tiles[ a ][ j ] ];
  }

  return q;
}

// Returns index (action) of largest entry in Q array, breaking ties randomly
int QLearningAgent::argmaxQ()
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

void QLearningAgent::updateWeights( double delta )
{
  double tmp = delta * alpha / numTilings;
  for ( int i = 0; i < numNonzeroTraces; i++ ) {
    int f = nonzeroTraces[ i ];
    if ( f > RL_MEMORY_SIZE || f < 0 )
      cerr << "f is too big or too small!!" << f << endl;
    weights[ f ] += tmp * traces[ f ];
  }
}

void QLearningAgent::loadTiles( double state[] )
{
  int tilingsPerGroup = 32;  /* num tilings per tiling group */
  numTilings = 0;

  /* These are the 'tiling groups'  --  play here with representations */
  /* One tiling for each state variable */
  for ( int v = 0; v < getNumFeatures(); v++ ) {
    for ( int a = 0; a < getNumActions(); a++ ) {
      GetTiles1( &(tiles[ a ][ numTilings ]), tilingsPerGroup, colTab,
		 state[ v ] / tileWidths[ v ], a , v );
    }
    numTilings += tilingsPerGroup;
  }
  if ( numTilings > RL_MAX_NUM_TILINGS )
    cerr << "TOO MANY TILINGS! " << numTilings << endl;
}

void QLearningAgent::setParams(int iCutoffEpisodes, int iStopLearningEpisodes)
{
  /* set learning parameters */
}
