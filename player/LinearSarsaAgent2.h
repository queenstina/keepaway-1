#ifndef LINEAR_SARSA_AGENT2
#define LINEAR_SARSA_AGENT2

#include "SMDPAgent.h"
#include "tiles2.h"

#define RL_MEMORY_SIZE 1048576
#define RL_MAX_NONZERO_TRACES 100000
#define RL_MAX_NUM_TILINGS 6000

class LinearSarsaAgent2:public SMDPAgent
{
  char weightsFile[256];
  bool bLearning;
  bool bSaveWeights;

  int epochNum;
  int lastAction;

  double alpha;
  double gamma;
  double lambda;
  double epsilon;

  double tileWidths[ MAX_STATE_VARS ];
  double Q[ MAX_ACTIONS ];

  double weights[ MAX_ACTIONS ][ RL_MEMORY_SIZE ];
  double traces[ MAX_ACTIONS ][ RL_MEMORY_SIZE ];

  int tiles[ RL_MAX_NUM_TILINGS ];
  int numTilings;

  double minimumTrace;
  int nonzeroTraces[ MAX_ACTIONS ][ RL_MAX_NONZERO_TRACES ];
  int numNonzeroTraces[ MAX_ACTIONS ];
  int nonzeroTracesInverse[ MAX_ACTIONS ][ RL_MEMORY_SIZE ];

  collision_table *colTab;

  // Load / Save weights from/to disk
  bool loadWeights( char *filename );
  bool saveWeights( char *filename );

  // Value function methods for CMACs
  int  selectAction();
  void initializeTileWidths( int numK, int numT );
  double computeQ( int a );
  int  argmaxQ();
  void updateWeights( double delta );
  void loadTiles( double state[] );

  // Eligibility trace methods
  void clearTrace( int a, int f );
  void clearExistentTrace( int a, int f, int loc );
  void decayTraces( int a,  double decayRate );
  void setTrace( int a, int f, float newTraceValue );
  void increaseMinTrace(int a);

 public:
  LinearSarsaAgent2                  ( int    numFeatures,
				      int    numActions,
				      bool   bLearn,
				      double widths[],
				      char   *loadWeightsFile,
				      char   *saveWeightsFile );

  // SMDP Sarsa implementation
  int  startEpisode( double state[] );
  int  step( double reward, double state[] );
  void endEpisode( double reward );
  void setParams(int iCutoffEpisodes, int iStopLearningEpisodes);
} ;

#endif
