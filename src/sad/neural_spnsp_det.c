#include<stdio.h>
#include<stdlib.h>
#include<math.h>
#include"neural_spnsp_incl.h"

#define mlpmemallocerr printf("Not able to allocate memory for MLP.\n");return(NULL);

int Neural_Spnsp_Det(
	float **feature,
	int dim,
	int context,
	MLP *mlp,
	float threshold,
	int *nsp_flag
)
{
 int i,j,k;
 float *mlpinfeat;
 int mlpinfeatdim;
 float *mlp_outpost;

 mlpinfeatdim=dim*(2*context+1);
 mlpinfeat=(float *)calloc(mlpinfeatdim,sizeof(float));
 if(mlpinfeat==NULL){printf("Not able to allocate memory at 'Neural_Spnsp_Det'.\n");return(-1);}

 mlp_outpost=(float *)calloc(mlp->nu_layer[mlp->nb_layer],sizeof(float));
 if(mlp_outpost==NULL){printf("Not able to allocate memory at 'Neural_Spnsp_Det'.\n");free(mlpinfeat);return(-1);}

 k=0;
 for(i=0;i<(2*context)+1;i++)
 {
  for(j=0;j<dim;j++)
  { mlpinfeat[k]=feature[i][j]; k++; }
 }

 Mlp_Post(mlpinfeatdim,mlpinfeat,mlp,mlp_outpost);

 if(mlp_outpost[0]>=threshold) { *nsp_flag=0; }
 else { *nsp_flag=1; }

 /*
 printf("MLP Output %f\n", mlp_outpost[0]);
 */

 free(mlp_outpost);
 free(mlpinfeat);

 return(0);
}

int Mlp_Post(
	int featdim,
	float *feature,
	MLP *mlp,
	float *out
)
{
 int i,j,k,l,h;

 for(i=0;i<mlp->nb_layer+1;i++)
 {
  for(j=0;j<mlp->nu_layer[i];j++)
    mlp->vlayer[i][j]=0.0;
 }

 for(i=0;i<mlp->nu_layer[0];i++)
  feature[i]=(feature[i]-mlp->mean[i])/mlp->var[i];

 Mlp_Ffwd(mlp,feature,/*mlp->outlayer_nonlinflag*/1,out,mlp->vlayer);

 return(0);
}

int Mlp_Ffwd(
	MLP *mlp,
	float *input,
	int outlayer_nonlinflag,
	float *output,
	float **vlayer
)
{
 int l,u1,u2;
 double tot;
 float temp;
 float sum;
 float max;

 for(u1=0;u1<mlp->nu_layer[0];u1++)
    *(*(vlayer)+u1) = *(input+u1); 

 for(l=1;l<mlp->nb_layer;l++)
 {
  for(u2=0;u2<mlp->nu_layer[l];u2++)
  {
   temp=mlp->bias[l-1][u2];
   for(u1=0;u1<mlp->nu_layer[l-1];u1++)
     temp+=mlp->w[l-1][u1][u2] * *(*(vlayer+(l-1))+u1);
   *(*(vlayer+l)+u2)=LSA(temp);
  }
 }

 for(u2=0;u2<mlp->nu_layer[mlp->nb_layer];u2++)
 {
  *(*(vlayer+mlp->nb_layer)+u2)=mlp->bias[mlp->nb_layer-1][u2];
  for(u1=0;u1<mlp->nu_layer[mlp->nb_layer-1];u1++)
    *(*(vlayer+mlp->nb_layer)+u2)+=mlp->w[mlp->nb_layer-1][u1][u2] * *(*(vlayer+(mlp->nb_layer-1))+u1);
 }

 if(outlayer_nonlinflag==0)
 {
  for(u2=0;u2<mlp->nu_layer[mlp->nb_layer];u2++)
    *(output+u2)=*(*(vlayer+mlp->nb_layer)+u2);
 }
 else
 {
  max=0.0;
  for(u2=0;u2<mlp->nu_layer[mlp->nb_layer];u2++)
  {
   if(*(*(vlayer+mlp->nb_layer)+u2)>max)
     max=*(*(vlayer+mlp->nb_layer)+u2);
  }
  tot=0.0;
  for(u2=0;u2<mlp->nu_layer[mlp->nb_layer];u2++)
    tot+=exp(*(*(vlayer+mlp->nb_layer)+u2)-max);
  for(u2=0;u2<mlp->nu_layer[mlp->nb_layer];u2++)
    *(output+u2)=exp(*(*(vlayer+mlp->nb_layer)+u2)-max)/tot;
 }

 return(0);
}

int Read_Mlp_Param(
	const char* file,
	MLP *mlp,
	int featdim,
	int context
)
{
 int i,j,k,l,u1,u2;
 int nwts,nbiases;
 char tempstr[100];
 int temp;
 float tempmean[1000];
 float tempvar[1000];

 FILE *fip;

 fip=fopen(file,"r");
 if(fip==NULL){printf("Not able to open the file '%s'.\n",file);return(-1);}
 
 for(l=0;l<mlp->nb_layer;l++)
 {
  if(fscanf(fip,"%s%d",tempstr,&nwts)<2){printf("Error reading '%s'.\n",file);fclose(fip);return(-1);}

  if( mlp->nu_layer[l]*mlp->nu_layer[l+1] != nwts ) {printf("Mismatch in file '%s'.\n",file);fclose(fip);return(0);}
  
  for(u2=0;u2<mlp->nu_layer[l+1];u2++)
  {
   for(u1=0;u1<mlp->nu_layer[l];u1++)
     if(fscanf(fip,"%f",&mlp->w[l][u1][u2])<1){printf("Error reading '%s'.\n",file);fclose(fip);return(-1);};
  }
 }

 for(l=0;l<mlp->nb_layer;l++)
 {
  if(fscanf(fip,"%s%d",tempstr,&nbiases)<2){printf("Error reading '%s'.\n",file);fclose(fip);return(-1);}
  if(mlp->nu_layer[l+1] != nbiases){printf("Mismatch in file '%s'.\n",file);fclose(fip);return(0);}
  for(u2=0;u2<mlp->nu_layer[l+1];u2++)
    if(fscanf(fip,"%f",&mlp->bias[l][u2])<1){printf("Error reading '%s'.\n",file);fclose(fip);return(-1);};
 }
 
 if(fscanf(fip,"%s%d",tempstr,&temp)<2){printf("Error reading '%s'.\n",file);fclose(fip);return(-1)
;}
 if(featdim != temp){printf("Mismatch in file '%s'.\n",file);fclose(fip);return(0);}

 for(i=0;i<featdim;i++)
   if(fscanf(fip,"%f",&tempmean[i])<1){printf("Error reading '%s'.\n",file);fclose(fip);return(-1);}; 

 if(fscanf(fip,"%s%d",tempstr,&temp)<2){printf("Error reading '%s'.\n",file);fclose(fip);return(-1)
;}
 if(featdim != temp){printf("Mismatch in file '%s'.\n",file);fclose(fip);return(0);}

 for(i=0;i<featdim;i++)
   if(fscanf(fip,"%f",&tempvar[i])<1){printf("Error reading '%s'.\n",file);fclose(fip);return(-1);};

 fclose(fip);

 k=0;
 for(i=0;i<2*context+1;i++)
 {
  for(j=0;j<featdim;j++)
  { mlp->mean[k]=tempmean[j]; mlp->var[k]=tempvar[j]; k++; }
 }

 return(0);
}

MLP *Mlp_Param_Mem_Alloc(
	int featdim,
	int context,
	int nhiddenunits,
	int noutputunits
)
{
 int i,j,k,l;
 MLP *mlp;
 int ninputunits;

 ninputunits=featdim*(2*context+1);

 mlp=(MLP *)calloc(1,sizeof(MLP));
 if(mlp==NULL) {mlpmemallocerr;}

 //mlp->nb_layer=nhiddenlayers+1;
 mlp->nb_layer=2;

 mlp->nu_layer=(int *)calloc(mlp->nb_layer+1,sizeof(int));
 if(mlp->nu_layer==NULL) {free(mlp);mlpmemallocerr;}

 mlp->nu_layer[0]=ninputunits;
 /*for(i=1;i<mlp->nb_layer;i++)
   mlp->nu_layer[i]=nhiddenunits[i-1];*/
 mlp->nu_layer[1]=nhiddenunits;
 mlp->nu_layer[mlp->nb_layer]=noutputunits;

 mlp->bias=(float **)calloc(mlp->nb_layer,sizeof(float *));
 if(mlp->bias==NULL) {free(mlp->nu_layer);free(mlp);mlpmemallocerr;}
 for(i=0;i<mlp->nb_layer;i++)
 {
  mlp->bias[i]=(float *)calloc(mlp->nu_layer[i+1],sizeof(float));
  if(mlp->bias[i]==NULL)
  {
   for(l=0;l<i;l++)
     free(mlp->bias[l]);
   free(mlp->bias);free(mlp->nu_layer);free(mlp);mlpmemallocerr;
  }
 }

 mlp->w=(float ***)calloc(mlp->nb_layer,sizeof(float **));
 if(mlp->w==NULL)
 {
  for(l=0;l<mlp->nb_layer;l++)
    free(mlp->bias[l]);
  free(mlp->bias);free(mlp->nu_layer);free(mlp);mlpmemallocerr;
 }
 for(i=0;i<mlp->nb_layer;i++)
 {
  mlp->w[i]=(float **)calloc(mlp->nu_layer[i],sizeof(float *));
  if(mlp->w[i]==NULL)
  {
   for(l=0;l<i;l++)
     free(mlp->w[l]);
   free(mlp->w);
   for(l=0;l<mlp->nb_layer;l++)
     free(mlp->bias[l]);
   free(mlp->bias);free(mlp->nu_layer);free(mlp);mlpmemallocerr;
  }
 }
 for(i=0;i<mlp->nb_layer;i++)
 {
  for(j=0;j<mlp->nu_layer[i];j++)
  {
   mlp->w[i][j]=(float *)calloc(mlp->nu_layer[i+1],sizeof(float));
   if(mlp->w[i][j]==NULL)
   {
    for(l=0;l<i;l++)
    {
     for(k=0;k<mlp->nu_layer[l];k++)
       free(mlp->w[l][k]);
    }
    for(k=0;k<j;k++)
      free(mlp->w[i][k]);
    for(l=0;l<mlp->nb_layer;l++)
    {
     free(mlp->w[l]);free(mlp->bias[l]);
    }
    free(mlp->w);free(mlp->bias);free(mlp->nu_layer);free(mlp);mlpmemallocerr;
   }
  }
 }

 mlp->mean=(float *)calloc(mlp->nu_layer[0],sizeof(float));
 if(mlp->mean==NULL)
 {
  for(l=0;l<mlp->nb_layer;l++)
  {
   for(k=0;k<mlp->nu_layer[l];k++)
     free(mlp->w[l][k]);
   free(mlp->w[l]);free(mlp->bias[l]);
  }
  free(mlp->w);free(mlp->bias);free(mlp->nu_layer);free(mlp);mlpmemallocerr;
 }

 mlp->var=(float *)calloc(mlp->nu_layer[0],sizeof(float));
 if(mlp->var==NULL)
 {
  free(mlp->mean);
  for(l=0;l<mlp->nb_layer;l++)
  {
   for(k=0;k<mlp->nu_layer[l];k++)
     free(mlp->w[l][k]);
   free(mlp->w[l]);free(mlp->bias[l]);
  }
  free(mlp->w);free(mlp->bias);free(mlp->nu_layer);free(mlp);mlpmemallocerr;
 }

 mlp->vlayer=(float **)calloc(mlp->nb_layer+1,sizeof(float *));
 if(mlp->vlayer==NULL)
 {
  free(mlp->var);
  free(mlp->mean);
  for(l=0;l<mlp->nb_layer;l++)
  {
   for(k=0;k<mlp->nu_layer[l];k++)
     free(mlp->w[l][k]);
   free(mlp->w[l]);free(mlp->bias[l]);
  }
  free(mlp->w);free(mlp->bias);free(mlp->nu_layer);free(mlp);mlpmemallocerr;
 }

 for(i=0;i<mlp->nb_layer+1;i++)
 {
  mlp->vlayer[i]=(float *)calloc(mlp->nu_layer[i],sizeof(float));
  if(mlp->vlayer[i]==NULL)
  {
   for(l=0;l<i;l++)
     free(mlp->vlayer[l]);
   free(mlp->vlayer);free(mlp->var);free(mlp->mean);
   for(l=0;l<mlp->nb_layer;l++)
   {
    for(k=0;k<mlp->nu_layer[l];k++)
      free(mlp->w[l][k]);
    free(mlp->w[l]);free(mlp->bias[l]);
   }
   free(mlp->w);free(mlp->bias);free(mlp->nu_layer);free(mlp);mlpmemallocerr;
  }
 } 

 return(mlp);
}

int Free_Mlp_Param_Mem(
	MLP *mlp
)
{
 int i,j,k;

 for(i=0;i<mlp->nb_layer;i++)
   free(mlp->vlayer[i]);
 free(mlp->vlayer);free(mlp->var);free(mlp->mean);
 for(i=0;i<mlp->nb_layer;i++)
 {
  for(j=0;j<mlp->nu_layer[i];j++)
    free(mlp->w[i][j]);
  free(mlp->w[i]);free(mlp->bias[i]);
 }
 free(mlp->w);free(mlp->bias);free(mlp->nu_layer);free(mlp);

 return(0);
}
