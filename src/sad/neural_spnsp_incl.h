#ifndef NEURAL_SPNSP_INCL_H
#define NEURAL_SPNSP_INCL_H

#ifdef __cplusplus
extern "C" {
#endif

#define hold printf("hp\n");fflush(stdin);getchar();
#define bpi(i) printf("bpi %d\n",i);fflush(stdin);getchar();
#define bpf(f) printf("bpf %f\n",f);fflush(stdin);getchar();
#define bpe(e) printf("bpe %e\n",e);fflush(stdin);getchar();
#define bps(s) printf("bps %s\n",s);fflush(stdin);getchar();
#define pvi(i) printf("%d\n",i);
#define pvf(f) printf("%f\n",f);
#define pve(e) printf("%e\n",e);
#define pvs(s) printf("%s\n",s);

#ifndef LSA    /*Logistic Sigmoid Activation */
# define  LSA(a)   1/(1+exp(-a))
#endif

#ifndef TANH   /* Tangent Hyperbolic */
# define  TANH(a)    (exp(a)-exp(-a))/(exp(a)+exp(-a))
#endif

#ifndef LSAD   /* Logistic Sigmoid Activation Derivative */
# define  LSAD(a)   a*(1-a)
#endif

#ifndef TANHD
# define   TANHD(a)   1-TANH(a)
#endif

typedef struct{
        int nb_layer;
        int *nu_layer;
        float ***w;
        float **bias;
	float *mean;
	float *var;
	int output_nonlinflag;
	float **vlayer;
}MLP;

MLP *Mlp_Param_Mem_Alloc(
        int featdim,
        int context,
        int nhiddenunits,
        int noutputunits
);

int Free_Mlp_Param_Mem(
        MLP *mlp
);

int Read_Mlp_Param(
        const char* file,
        MLP *mlp,
        int featdim,
        int context
);

int Neural_Spnsp_Det(
        float **feature,
        int dim,
        int context,
        MLP *mlp,
        float threshold,
        int *nsp_flag
);

int Mlp_Post(
        int featdim,
        float *feature,
        MLP *mlp,
        float *out
);

int Mlp_Ffwd(
        MLP *mlp,
        float *input,
        int outlayer_nonlinflag,
        float *out,
        float **vlayer
);

#ifdef __cplusplus
}
#endif

#endif
