/* Retained-matrix native PARDISO ABI prerequisite. GPL-2.0-or-later. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
extern void pardiso_main(double*,double*,double*,double*,double*,double*,
                         int32_t*,int32_t*,int32_t*,int32_t*,int32_t*,
                         int32_t*,int32_t*,int32_t*,int32_t*);
static void readn(FILE *f,void *p,size_t s,size_t n){
 if(fread(p,s,n,f)!=n)exit(2);
}
int main(int argc,char **argv){
 if(argc!=3)return 1;
 FILE *f=fopen(argv[1],"rb");if(!f)return 2;
 int32_t n,nz;readn(f,&n,4,1);readn(f,&nz,4,1);
 if(n<=0||nz<=0)return 3;
 int32_t *jq=malloc(4*((size_t)n+1)),*ir=malloc(4*(size_t)nz),*icol=malloc(4*(size_t)n);
 double *ad=malloc(8*(size_t)n),*au=malloc(16*(size_t)nz),*b=malloc(8*(size_t)n);
 if(!jq||!ir||!icol||!ad||!au||!b)return 4;
 readn(f,jq,4,n+1);readn(f,ir,4,nz);readn(f,ad,8,n);readn(f,au,8,2*(size_t)nz);readn(f,b,8,n);fclose(f);
 for(int32_t i=0;i<n;i++)icol[i]=jq[i+1]-jq[i];
 int32_t symmetry=2,format=1,nrhs=1;double sigma=0;
 pardiso_main(ad,au,NULL,NULL,&sigma,b,icol,ir,&n,&nz,&symmetry,&format,jq,&nz,&nrhs);
 f=fopen(argv[2],"wb");if(!f)return 5;
 if(fwrite(b,8,n,f)!=(size_t)n)return 6;
 fclose(f);free(jq);free(ir);free(icol);free(ad);free(au);free(b);return 0;
}
