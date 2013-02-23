#include<stdio.h>
#include<stdlib.h>
#include<tiles2.h>
#include<math.h>

#define DIMENSION 2
#define MEMORY_SIZE 4096
#define ACTIONS 4
#define TILINGS 32
#define SCALE 0.25

float randf(){
	return (float)rand()/(float)RAND_MAX;
}

void setWeight(float weights[][MEMORY_SIZE], float value){
		for (int j=0; j<ACTIONS; j++)
			for (int i=0; i<MEMORY_SIZE; i++){
				weights[j][i]=value;
			}
}


//Funcoes para simular o ambiente
float x,y;
void initEpisode(){
	x = 0.0;
	y = 0.0;
}

bool endEpisode(){
	if ( (x>=.9) & (y>=.9) ) return true;
	else return false;
}

void observe(float variables[]){
	variables[0] = x;
	variables[1] = y;
}

float execute(int action){
	if ((action < 0) | (action > ACTIONS-1))
		printf("BUG: invalid action!\n");
		
	if ( endEpisode() )	
		printf("BUG: episode already ended!\n");

	switch (action){
		case 0: //Norte
			y += 0.05*randf();
			if (y>1) y = 1;
			break;
		case 1: //Sul
			y -= 0.05*randf();
			if (y<0) y = 0;
			break;
		case 2: //Leste
			x += 0.05*randf();
			if (x>1) x = 1;
			break;
		case 3: //Oeste
			x -= 0.05*randf();
			if (x<0) x = 0;
			break;
	}
	
	if ( (x>=.9) & (y>=.9)) return 1;
	else return 0;
}

// Funcao que sorteia uma acao segundo a politica estocastica
int chooseAction(int tiles_array[], float politica[][MEMORY_SIZE], float *prob, float epsilon=0.0){
	float results[ACTIONS];
	float total = 0;

	for (int j=0; j<ACTIONS; j++){
		results[j] = 0;
		for (int i=0;i<TILINGS;i++)
		 	results[j] += politica[j][tiles_array[i]];
      total += results[j];
	}
					
	if ((total>1.01) | (total < 0.99)) printf("BUG: soma de probabilidades inconsistente = %f\n",total);
						
	int a = -1;
	float aux = randf();
	float accum = 0;
	for (int j=0; (j<ACTIONS) & (a < 0); j++)
		if (results[j]+accum > aux-0.001) a = j;  // subtrai 0.001 aqui para nao ter problema com probabilidades proximo de 1
		else accum += results[j];

	if (randf() > (1-epsilon)) a = rand()%ACTIONS;

	prob[0] = (1-epsilon)*results[a] + epsilon/ACTIONS;
	return a;
}

int main()
{

	float vars_array[DIMENSION]; // Variavel para guardar as coordenadas da observação

	float gradienteLocal[ACTIONS][MEMORY_SIZE]; // Pesos do gradiente sendo calculado em um determinado periodo
	float gradienteGlobal[ACTIONS][MEMORY_SIZE]; // Pesos do gradiente utilizado para atualizar a política. Este gradiente e' atualizado com o gradiente anterior
	float politica[ACTIONS][MEMORY_SIZE]; // Pesos da politica estocastica
	float eligibility[ACTIONS][MEMORY_SIZE]; // Pesos do traco de eligibilidade por episodio
	
	int tiles_array[TILINGS]; // Variavel para guardar a descricao discreta da observacao
	
	float epsilon = 0.1; // Valor que controla o limite greedy para a politica estocastica
	float alphaGrad = 0.1; // taxa de aprendizado para o gradiente global
	float gamma = 0.99; // fator de desconto
	float delta = 0.1; // tamanho do passo utilizado para atualizar política
	
//	int episodes[40] = {250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250};
	int episodes[48] = {50,50,50,50,50,50,50,50,50,50,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250,250};
//	int episodes[29] = {50,50,50,50,50,50,50,50,50,50,100,100,100,100,100,250,250,250,250,500,500,500,500,1000,1000,1000,1000,1000,1000};
	int nRuns = 100;
	int nPeriodos = 48;
	int nEpisodes = 1000;
	int nTempo = 1000;


	FILE *file;
   if (!(file = fopen("grad.txt","w"))){
  		printf("Erro! Impossivel abrir o arquivo!\n");
  		exit(1);
  	}else{
  		printf("Gravando resultados no arquivo 'grad.txt'!\n");
  	}
   

	// Inicia uma nova execucao com todo conhecimento zerado;
	for (int run=0; run<nRuns; run++){

		setWeight(politica,1.0/(ACTIONS*TILINGS)); //isso inicializa a politica com uma distribuicao uniforme (note que deve-se dividir por TILINGS para garantir soma 1)
		setWeight(gradienteGlobal,0.0);
		

		// Inicia um novo periodo para estimativa do gradiente local
		for (int periodo=0; periodo<nPeriodos;periodo++){
			setWeight(gradienteLocal,0.0);

			// Inicia um novo episodio
			for (int episode=0; episode<episodes[periodo]; episode++){
				setWeight(eligibility,0.0);
				
				initEpisode();

				int tempo = 0;																				

				while ( !endEpisode() & (tempo < nTempo) ){
				
					observe(vars_array);

					vars_array[0] = vars_array[0] / SCALE;
					vars_array[1] = vars_array[1] / SCALE;

					GetTiles(tiles_array,TILINGS,MEMORY_SIZE,vars_array,DIMENSION); // pega o estado discreto do tiling code

					float prob;
					int action = chooseAction(tiles_array,politica,&prob,epsilon); // pega também a probalidade de execucao da acao escolhido para atualizar traco de eligibilidade
					float r = execute(action);

					// atualiza gradiente local (isso so' faz sentido se recompensa for diferente de 0
					if (r != 0){
						for (int j=0; j<ACTIONS; j++)
							for (int i=0; i<MEMORY_SIZE; i++){
								gradienteLocal[j][i] += eligibility[j][i]*r*powf(gamma,tempo)/episodes[periodo];
							}
					}

					// atualiza traco de eligibilidade
					for (int i=0;i<TILINGS;i++) {
						eligibility[action][tiles_array[i]] += 1/((1-epsilon)*prob + epsilon/ACTIONS);
					}

					tempo++;			
				} // Fim de episodio
				fprintf(file,"%d ",tempo);
			} // Fim de periodo
			

			// atualiza gradiente global com o gradiente local
			for (int j=0; j<ACTIONS; j++)
				for (int i=0; i<MEMORY_SIZE; i++)
					gradienteGlobal[j][i] += alphaGrad*(gradienteLocal[j][i] - gradienteGlobal[j][i]);
		
			// atualiza a política estocastica (aqui e' feito para todos os tiles, mas podemos testar para apenas os tiles visitados)
			for (int i=0; i<MEMORY_SIZE; i++){
				// encontra a melhor politica
				int aStar = 0;
				float maxCur = gradienteGlobal[0][i];
				for (int j=1; j<ACTIONS; j++)
					if (gradienteGlobal[j][i]>maxCur){
						aStar = j;
						maxCur = gradienteGlobal[j][i];
					}
			
				// atualiza a política (a divisao por TILINGS garante que a somatoria continua 1)
				for (int j=0; j<ACTIONS; j++){
					if (aStar == j) politica[j][i] = (1-delta)*politica[j][i] + delta/TILINGS;
					else politica[j][i] = (1-delta)*politica[j][i];
				}
			}
			
		} // Fim de execucao
		fprintf(file,"\n");
		printf("%1.2f ",float(run+1)/nRuns);fflush(stdout);
		
		

	}
	printf("\n");
	fclose(file);
	return(0);
}
