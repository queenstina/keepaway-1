#include "AbsProbAgent.h"

AbsProbAgent::AbsProbAgent(int numFeatures, int numActions, bool bLearn, double widths[], char *loadWeightsFile, char *saveWeightsFile ) :
    SMDPAgent(numFeatures, numActions)
{
    agentId = getpid() % 10;
    cout << "Agente " << agentId << " inicializado. v1.7" << endl;
    this->epsilon = 0.01; // Valor que controla o limite greedy para a politica estocastica
    this->gamma = 0.99; // fator de desconto
    this->delta = 0.1; // tamanho do passo utilizado para atualizar política
    this->colTab = new collision_table( MEMORY_SIZE, 1 );
    int tmp[2];
    float tmpf[2];
    GetTiles( tmp, 1, 1, tmpf, 0 );  // A dummy call to set the hashing table

    for (int i = 0; i < getNumFeatures(); i++) {
        tileWidths.push_back(widths[i]);
    }
    numTilings = numFeatures * TILINGS_PER_GROUP;
    tiles = (int*) malloc(numTilings * sizeof(int));
    
    gradienteLocal.resize(numActions);
    gradienteGlobal.resize(numActions);
    politica.resize(numActions);
    eligibility.resize(numActions);
    for(int a=0; a < numActions; a++) {
        gradienteLocal[a].resize(MEMORY_SIZE);
        gradienteGlobal[a].resize(MEMORY_SIZE);
        politica[a].resize(MEMORY_SIZE);
        eligibility[a].resize(MEMORY_SIZE);
    }
    
    periodo = 0;
    episodio = 0;
    passo = 0;
    lastAction = -1;
    prob = 1.0;

    // Inicia uma nova execucao com todo conhecimento zerado;
    setValues(politica,1.0/(numActions*numTilings)); //isso inicializa a politica com uma distribuicao uniforme (note que deve-se dividir por TILINGS para garantir soma 1)
    setValues(gradienteGlobal, 0.0);

    episodioGlobal = 0;
    rewardAccum = 0;

    ostringstream ss;
    ss << "gradiente" << agentId << ".txt";
    file.open(ss.str().c_str());
    if (file.is_open()) {
        cout << "Arquivo aberto com sucesso!" << endl;
    } else {
        cout << "Erro ao abrir arquivo." << endl;
    }
    ostringstream ss2;
    ss2 << "politica" << agentId << ".txt";
    file2.open(ss2.str().c_str());
}

AbsProbAgent::~AbsProbAgent()
{
    file.close();
    file2.close();
}

void AbsProbAgent::startPeriodo() {
    periodo++;
    setValues(gradienteLocal, 0.0);
    //cout << "Periodo: " << periodo << endl;
}

int  AbsProbAgent::startEpisode(double state[] ) {

    if (episodio == 0) {
        startPeriodo();
    }

    episodio++;
    episodioGlobal++;
    passo = 0;
    setValues(eligibility, 0.0);
    rewardEpisode = 0;

    loadTiles(state);
    // pega também a probalidade de execucao da acao escolhido para atualizar traco de eligibilidade
    lastAction = chooseAction();

    return lastAction;
}

int  AbsProbAgent::step(double reward, double state[] ) {
    
    passo++;

    // atualiza gradiente local (isso so' faz sentido se recompensa for diferente de 0
    updateGradienteLocal(reward);
    
    // atualiza traco de eligibilidade
    updateTrace(lastAction);

    loadTiles(state);

    // pega também a probalidade de execucao da acao escolhido para atualizar traco de eligibilidade
    lastAction = chooseAction();
    
    if (!isnan(reward) && reward > 0 && reward < 500) rewardEpisode += reward;

    return lastAction;
}

void AbsProbAgent::endEpisode( double reward ) {
    
    // atualiza gradiente local (isso so' faz sentido se recompensa for diferente de 0
    updateGradienteLocal(reward);
    
    if (!isnan(reward) && reward > 0 && reward < 500) rewardEpisode += reward;
    rewardAccum += (rewardEpisode - rewardAccum) / (double) episodioGlobal;
    //cout << agentId << "- Reward episode " << episodioGlobal << ": " << rewardEpisode << endl;

    if (episodio == getNumEpisodiosPeriodo()) {
        endPeriodo();
    }
    visitedTilesEpisode.clear();
}

void AbsProbAgent::endPeriodo() {
    episodio = 0;

    cout << agentId << " - Fim periodo " << periodo << endl;

    for (int i=0; i < MEMORY_SIZE; i++) {
        for (int j=0; j < getNumActions(); j++) {
        //for (unsigned int k=0; k < visitedTilesPeriod.size(); k++) {
            //int i = visitedTilesPeriod[k];
            gradienteGlobal[j][i] += getAlphaGrad()*(gradienteLocal[j][i] - gradienteGlobal[j][i]);         
        }
    }

    // atualiza a política estocastica (aqui e' feito para todos os tiles, mas podemos testar para apenas os tiles visitados)
    for (int i=0; i < MEMORY_SIZE; i++) {
    //for (unsigned int k=0; k < visitedTilesPeriod.size(); k++) {
        //int i = visitedTilesPeriod[k];
        // encontra a melhor politica
        int aStar = 0;
        double maxCur = gradienteGlobal[0][i];
        for (int j=1; j < getNumActions(); j++)
            if (gradienteGlobal[j][i] > maxCur) {
                aStar = j;
                maxCur = gradienteGlobal[j][i];
            }

        // atualiza a política (a divisao por TILINGS garante que a somatoria continua 1)
        for (int j=0; j < getNumActions(); j++) {
            if (aStar == j) politica[j][i] = (1-delta)*politica[j][i] + delta/numTilings;
            else politica[j][i] = (1-delta)*politica[j][i];
        }
    }

    visitedTilesPeriod.clear();
    cout << agentId << "- Reward medio: " << rewardAccum << endl;

}

void AbsProbAgent::setParams(int iCutoffEpisodes, int iStopLearningEpisodes) {
    // Nada
}

void AbsProbAgent::setValues(vector< vector<double> > array, double value) {
    /// Atribui valores a todos elementos de um array
    for (unsigned int i=0; i < array.size(); i++) {        
        array[i].assign(array[i].size(), value);
    }
    cout << "setValues: " << array[1][1] << endl;
}

void AbsProbAgent::loadTiles( double state[] ) {

  int idxTilings = 0;

  /* One tiling for each state variable */
  for ( int v = 0; v < getNumFeatures(); v++ ) {
      GetTiles1( &(tiles[idxTilings]), TILINGS_PER_GROUP, colTab, state[v]/tileWidths[v], v);
      idxTilings += TILINGS_PER_GROUP;
  }
  if ( idxTilings > RL_MAX_NUM_TILINGS )
    cerr << "TOO MANY TILINGS! " << idxTilings << endl;

}

// Funcao que sorteia uma acao segundo a politica estocastica
int AbsProbAgent::chooseAction()
{
    double results[getNumActions()];
    double total = 0;

    for (int j=0; j < getNumActions(); j++) {
        results[j] = 0;
        for (int i=0; i < numTilings; i++) {
            results[j] += politica[j][tiles[i]];
        }
      total += results[j];
    }

    if ((total>1.00001) | (total < 0.99999)) printf("BUG: soma de probabilidades inconsistente = %f\n",total);

    int a = -1;
    double aux = randf();
    double accum = 0;
    for (int j=0; (j < getNumActions()) & (a < 0); j++) {
        if (results[j]+accum > aux-0.001) a = j;  // subtrai 0.001 aqui para nao ter problema com probabilidades proximo de 1
        else accum += results[j];
    }

    if (randf() > (1-epsilon)) a = rand() % getNumActions();

    prob = (1-epsilon)*results[a] + epsilon/getNumActions();
    return a;
}

void AbsProbAgent::updateTrace(int action) {
    for (int i=0; i < numTilings; i++) {
        eligibility[action][tiles[i]] += 1.0/prob;
        for (int a = 0; a < getNumActions(); a++) {
            eligibility[a][tiles[i]] -= 1;
        }
        visitedTilesEpisode.push_back(tiles[i]);
        visitedTilesPeriod.push_back(tiles[i]);
    }
}

void AbsProbAgent::updateGradienteLocal(double reward) {
    if (!isnan(reward) && reward != 0) {
        for (int j=0; j < getNumActions(); j++) {
            for (int i=0; i < MEMORY_SIZE; i++) {
            //for (unsigned int k=0; k < visitedTilesEpisode.size(); k++) {
                //int i = visitedTilesEpisode[k];
                gradienteLocal[j][i] += eligibility[j][i]*reward*powf(gamma,passo) / getNumEpisodiosPeriodo();
            }
        }
    }
}

int AbsProbAgent::getNumEpisodiosPeriodo() {
    return 1000;
    //return 20 * periodo;
    //return 5 + periodo;
}

double AbsProbAgent::getAlphaGrad() {
    return 0.1;
    //return periodo > 0 ? 1.0 / periodo : 1.0;
}
