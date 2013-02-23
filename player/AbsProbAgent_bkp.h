#ifndef ABSPROBAGENT_H
#define ABSPROBAGENT_H

#include "SMDPAgent.h"
#include "tiles2.h"
#include <cmath>
#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>

//#define MEMORY_SIZE 1048576
#define MEMORY_SIZE 262144
//#define MEMORY_SIZE 4096
#define RL_MAX_NUM_TILINGS 6000
#define TILINGS_PER_GROUP 32

using namespace std;

class AbsProbAgent : public SMDPAgent
{
    public:
        AbsProbAgent( int    numFeatures,
				          int    numActions,
				          bool   bLearn,
				          double widths[],
				          char   *loadWeightsFile,
                          char   *saveWeightsFile );
        virtual ~AbsProbAgent();

        // SMDP Sarsa implementation
        int  startEpisode( double state[] );
        int  step( double reward, double state[] );
        void endEpisode( double reward );
        void setParams(int iCutoffEpisodes, int iStopLearningEpisodes);

        int getNumEpisodiosPeriodo();
        double getAlphaGrad(); // taxa de aprendizado para o gradiente global


    protected:
    private:

        int agentId;
        ofstream file;

        int passo;
        int episodio;
        int periodo;

        int lastAction;
        double prob;

        double epsilon; // Valor que controla o limite greedy para a politica estocastica
	    double gamma; // fator de desconto
	    double delta; // tamanho do passo utilizado para atualizar política

        int episodioGlobal;
        double rewardEpisode;
	    double rewardAccum; // só pra saber se está evoluindo

	    double gradienteLocal[MAX_ACTIONS][MEMORY_SIZE]; // Pesos do gradiente sendo calculado em um determinado periodo
	    double gradienteGlobal[MAX_ACTIONS][MEMORY_SIZE]; // Pesos do gradiente utilizado para atualizar a política. Este gradiente e' atualizado com o gradiente anterior
	    double politica[MAX_ACTIONS][MEMORY_SIZE]; // Pesos da politica estocastica
	    double eligibility[MAX_ACTIONS][MEMORY_SIZE]; // Pesos do traco de eligibilidade por episodio
	    vector<int> visitedTilesEpisode;
	    vector<int> visitedTilesPeriod;

        int numTilings;
	    int* tiles; // Variavel para guardar a descricao discreta da observacao
	    double tileWidths[ MAX_STATE_VARS ];

	    collision_table *colTab;

	    void setValues(double array[][MEMORY_SIZE], double value);
	    void loadTiles( double state[] );

	    void startPeriodo();
	    void endPeriodo();

	    double randf() {
	        return (double)rand()/(double)RAND_MAX;
	    }

	    // Funcao que sorteia uma acao segundo a politica estocastica
        int chooseAction();
        void updateTrace(int action);
        void updatePolicy();
        void updateGradienteLocal(double reward);
};

#endif // ABSPROBAGENT_H
