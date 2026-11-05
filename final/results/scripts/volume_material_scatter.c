/* Isolated additive material-correction ABI prototype. GPL-2.0-or-later.
 * Uses unchanged CalculiX add_sm_st_as object; never a production solver.
 * Input/output are little-endian native int32/float64 on this guest. */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
extern void add_sm_st_as_(double*,double*,int32_t*,int32_t*,int32_t*,int32_t*,double*,int32_t*,int32_t*,int32_t*);
static void readn(void *p,size_t s,size_t n){if(fread(p,s,n,stdin)!=n)exit(2);}
int main(void){
 int32_t n,nz,count,enabled; readn(&n,4,1);readn(&nz,4,1);readn(&count,4,1);readn(&enabled,4,1);
 if(n<=0||nz<=0||count<0||(enabled!=0&&enabled!=1))return 3;
 int32_t *jq=malloc(4*(n+1)),*ir=malloc(4*nz); double *ad=malloc(8*n),*au=malloc(16*nz);
 if(!jq||!ir||!ad||!au)return 4;
 readn(jq,4,n+1);readn(ir,4,nz);readn(ad,8,n);readn(au,8,nz);
 /* Same lower-to-upper copy ordering as mafillsmasmain.c, before addition. */
 memcpy(au+nz,au,8*nz); int32_t ns[3]={nz,nz,nz},neutral=0;
 for(int32_t k=0;k<count;k++){
  int32_t i,j; double value;readn(&i,4,1);readn(&j,4,1);readn(&value,8,1);
  if(i<1||i>n||j<1||j>n)return 5;
  if(enabled)add_sm_st_as_(au,ad,jq,ir,&i,&j,&value,&neutral,&neutral,ns);
 }
 if(fwrite(ad,8,n,stdout)!=(size_t)n||fwrite(au,8,2*nz,stdout)!=(size_t)(2*nz))return 6;
 free(jq);free(ir);free(ad);free(au);return 0;
}
