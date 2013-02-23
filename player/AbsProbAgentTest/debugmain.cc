#include "../AbsProbAgent.h"

#include <iostream>
#include <fstream>

using namespace std;

int main() {

    cout << "Iniciando debug..." << endl;

    int numFeatures = 13;
    int numActions = 3;
    double widths[] = {2, 2, 3, 2, 3, 2, 3, 3, 3, 4, 4, 10, 10};

    ifstream ifs("sim4001.txt");

    double reward;
    double state[13];

    char s[100];

    AbsProbAgent agent(numFeatures, numActions, true, widths, s, s);


    for (int i = 0; i < numFeatures; i++) {
        ifs >> state[i];
    }

    while (ifs.good()) {

        agent.startEpisode(state);
        for (int j = 0; j < 5; j++) {
            ifs >> reward;
            for (int i = 0; i < numFeatures; i++) {
               ifs >> state[i];
            }
            agent.step(reward, state);
        }
        agent.endEpisode(reward);
    }

    ifs.close();

}
