#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>

#define THREAD_PER_BLOCK 16 // on fixe le nombre de colonnes à 16
#define COLUMNS 16



//fct gpu

__global__ void multiplication_matrix_GPU(int *a, int *b, int*c)
{
	int idx = blockIdx.x * THREAD_PER_BLOCK + threadIdx.x;

	int sum = 0;

        __shared__ int bs[16]; // définition de la mémoire partagée
 
        bs[threadIdx.x]= b[threadIdx.x];


	//for(int j = 0; j<COLUMNS;++j,++a,++b)
	//	sum += (a[idx*COLUMNS+j])*(b[j]);

        sum+= (a[idx*COLUMNS+0])*(bs[0]);
        sum+= (a[idx*COLUMNS+1])*(bs[1]);
        sum+= (a[idx*COLUMNS+2])*(bs[2]);
        sum+= (a[idx*COLUMNS+3])*(bs[3]);
        sum+= (a[idx*COLUMNS+4])*(bs[4]);
        sum+= (a[idx*COLUMNS+5])*(bs[5]);
        sum+= (a[idx*COLUMNS+6])*(bs[6]);
        sum+= (a[idx*COLUMNS+7])*(bs[7]);
        sum+= (a[idx*COLUMNS+8])*(bs[8]);
        sum+= (a[idx*COLUMNS+9])*(bs[9]);
        sum+= (a[idx*COLUMNS+10])*(bs[10]);
        sum+= (a[idx*COLUMNS+11])*(bs[11]);
        sum+= (a[idx*COLUMNS+12])*(bs[12]);
        sum+= (a[idx*COLUMNS+13])*(bs[13]);
        sum+= (a[idx*COLUMNS+14])*(bs[14]);
        sum+= (a[idx*COLUMNS+15])*(bs[15]);


	c[idx]=sum;
	__syncthreads();

}


int main(int agrc, char * argv[])
{
	unsigned int rows = atoi(argv[1]), i, j; // il y a un malloc contenant ligne et colonnes --> Matrice A et un malloc contenant que colonne -> vecteur B
	int * a_h = (int *) malloc(rows * COLUMNS * sizeof(int)), * b_h = (int *) malloc(COLUMNS * sizeof(int)), * c1_h = (int *) malloc(rows * sizeof(int)), * c2_h = (int *) malloc(rows * sizeof(int));

int *a_d, *b_d, *c_d;

//allocation sur GPU

cudaSetDevice (0);

cudaMalloc ((void**) &a_d , rows * COLUMNS * sizeof(int));
cudaMalloc ((void**) &b_d , COLUMNS * sizeof(int));
cudaMalloc ((void**) &c_d , rows * sizeof(int));


//copie vers GPU

cudaMemcpy (a_d , a_h , rows * COLUMNS *sizeof(int), cudaMemcpyHostToDevice ); // on copie les données du CPU vers le GPU
cudaMemcpy (b_d , b_h , COLUMNS * sizeof(int), cudaMemcpyHostToDevice );


	unsigned long long ref1, ref2;
	unsigned long long diffH = 0, diffD = 0;
	struct timeval tim;
	
	//remplissage de la matrice

	for(i=0;i<COLUMNS*rows;++i){
		if(i<COLUMNS){
			b_h[i] = i+1;
		}
		a_h[i] = rand()%(COLUMNS*rows);
	}
	
	//multiplication sur CPU

	gettimeofday(&tim, NULL);
	ref1 = tim.tv_sec * 1000000L + tim.tv_usec;
	int * a = a_h, *b, *c=c1_h;
	for(i = 0; i<rows; ++i){
		c1_h[i] = 0;
		for(j = 0; j<COLUMNS;++j,++a,++b)
			c1_h[i] += (a_h[i*COLUMNS+j])*(b_h[j]);
	}
	gettimeofday(&tim, NULL);
	ref2 = tim.tv_sec * 1000000L + tim.tv_usec;
  	diffH+=ref2-ref1; // différence des timing
	
	//multiplication sur GPU

  	gettimeofday(&tim, NULL);
	ref1 = tim.tv_sec * 1000000L + tim.tv_usec;
	
	// EXECUTION GPU  c'est ici que nous allons travailler
	
        int blocks = rows/THREAD_PER_BLOCK;
	multiplication_matrix_GPU<<<blocks,THREAD_PER_BLOCK>>>(a_d, b_d, c_d);
	
	
	
  	gettimeofday(&tim, NULL);
  	ref2 = tim.tv_sec * 1000000L + tim.tv_usec;
  	diffD+=ref2-ref1;

       cudaMemcpy(c2_h , c_d , rows * sizeof(int), cudaMemcpyDeviceToHost);
  	
	//vérification des résultats et nettoyage
  	int ok = 1;
  	for(i=0;i<10;++i)
  		if(c1_h[i]!=c2_h[i]){
  			//ok = 0;
  			printf("Différence : %d != %d\n", c1_h[i], c2_h[i]);
  		}
  	if(ok)
		printf("Temps de calcul, CPU [%llu usec] GPU [%llu usec] \n", diffH, diffD);
  	
        
	cudaFree(a_d);
	cudaFree(b_d);
	cudaFree(c_d);

  	free(a_h);
  	free(b_h);
  	free(c1_h);
  	free(c2_h);

}
