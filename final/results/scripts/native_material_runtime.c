/* Experimental integration only. GPL-2.0-or-later.
 * Separate cache follows native xstiff lifetime; matching geometry AND both state
 * arrays are mandatory before scatter. No constitutive/residual state is written.
 * Native results workers own disjoint element/IP slots. Assembly is serial.
 */
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
extern int thesis_material_delta(int,const double*,const double*,double,double,double,double,int,const double*,const double*,double*);
extern void shape20h_(double*,double*,double*,double*,double*,double*,int32_t*);
extern void add_sm_st_as_(double*,double*,int32_t*,int32_t*,int32_t*,int32_t*,double*,int32_t*,int32_t*,int32_t*);
typedef struct {int valid,plastic,ns; double strain[6],old[13],now[13],delta[36];} Slot;
static Slot *slots;static int elements;
int thesis_material_enabled(void){const char *v=getenv("THESIS_VOLUME_MATERIAL");return v && strcmp(v,"1")==0;}
static void fail(const char *s){fprintf(stderr,"THESIS_VOLUME_MATERIAL rejected: %s\n",s);fflush(stderr);exit(86);}
void thesis_material_prepare(int ne,int ips,int method,int mpc,int thermal,int contact,int trans,int restr){
 if(!thesis_material_enabled())return;
 if(ne<=0||ips!=8||method!=1||mpc||thermal||contact||trans||restr)fail("unsupported model scope");
 free(slots);slots=calloc((size_t)ne*8,sizeof(Slot));if(!slots)fail("allocation");elements=ne;
 fprintf(stderr,"THESIS_VOLUME_MATERIAL cache initialized %d elements\n",ne);
}
/* Called only where native resultsmech saves xstiff, after mechmodel returns. */
void thesis_material_capture_(int32_t *e,int32_t *ip,int32_t *kode,double *strain,double *elastic,double *table,double *old,double *now,int32_t *ns,int32_t *ielas,int32_t *orientation){
 if(!thesis_material_enabled())return;
 if(!slots||*e<1||*e>elements||*ip<1||*ip>8||*orientation)fail("capture index/orientation");
 Slot *s=slots+8*(*e-1)+*ip-1;s->valid=0;s->ns=0;s->plastic=0;
 memcpy(s->strain,strain,48);memset(s->delta,0,sizeof(s->delta));
 /* All nonplastic laws are deliberately rejected, except isotropic elasticity
  * (kode 2). An elastic first redistribution retains its native tangent. */
 if(*kode!=2 && *kode!=-51)fail("unsupported constitutive law");
 if(*kode==-51){
  if(*ns<13||*ns>13)fail("unsupported plastic state size");
  s->ns=13;memcpy(s->old,old,104);memcpy(s->now,now,104);
  for(int k=7;k<13;k++)if(fabs(old[k])>1e-14)fail("backstress");
  int nt=(int)table[800];if(nt<2||nt>200||(int)table[801]!=0)fail("hardening scope");
  double c[9]={1+2*strain[0],2*strain[3],2*strain[4],2*strain[3],1+2*strain[1],2*strain[5],2*strain[4],2*strain[5],1+2*strain[2]};
  double m[9]={1-2*old[1],-2*old[4],-2*old[5],-2*old[4],1-2*old[2],-2*old[6],-2*old[5],-2*old[6],1-2*old[3]};
  double ep[200],ys[200];for(int k=0;k<nt;k++){ep[k]=table[2*k];ys[k]=table[2*k+1];}
  if(!*ielas){
   if(thesis_material_delta(1,c,m,old[0],now[0],elastic[0]/(2*(1+elastic[1])),elastic[0]/(3*(1-2*elastic[1])),nt,ep,ys,s->delta))fail("material kernel");
   s->plastic=1;
  }
 }
 s->valid=*ielas ? 2 : 1;
}
static int closeval(double a,double b){return isfinite(a)&&isfinite(b)&&fabs(a-b)<=2e-11*fmax(1.,fmax(fabs(a),fabs(b)));}
void thesis_material_scatter(int ne,const double *co,const double *u,const int32_t *kon,const int32_t *ipkon,const char *lakon,const int32_t *mi,const double *old,const double *now,int ns,const int32_t *map,double *ad,double *au,int32_t *jq,int32_t *ir,int32_t *nzs){
 if(!thesis_material_enabled())return;
 if(ne!=elements||mi[0]!=8||mi[1]!=3)fail("assembly dimensions");
 const double a=0.57735026918962576451;
 const int signs[8][3]={{-1,-1,-1},{1,-1,-1},{-1,1,-1},{1,1,-1},{-1,-1,1},{1,-1,1},{-1,1,1},{1,1,1}};
 int32_t flag=3,neutral=0;int count=0;
 for(int e=0;e<ne;e++){
  if(ipkon[e]<0||memcmp(lakon+8*e,"C3D20R",6))fail("only active C3D20R supported");
  double xl[60],ul[60],K[3600]={0};int32_t eq[60];
  for(int n=0;n<20;n++){int node=kon[ipkon[e]+n]-1;for(int d=0;d<3;d++){xl[3*n+d]=co[3*node+d];ul[3*n+d]=u[4*node+d+1];eq[3*n+d]=map[4*node+d+1];}}
  for(int ip=0;ip<8;ip++){
   Slot *s=slots+8*e+ip;if(!s->valid)fail("missing matching native stiffness capture");
   if(s->valid==2)continue;
   if(s->ns){if(ns!=s->ns)fail("state dimension changed");size_t off=((size_t)e*8+ip)*ns;for(int k=0;k<ns;k++)if(old[off+k]!=s->old[k]||now[off+k]!=s->now[k])fail("stale previous/current state");}
   double xi=a*signs[ip][0],et=a*signs[ip][1],ze=a*signs[ip][2],det,shp[80],F[9]={1,0,0,0,1,0,0,0,1},c[9]={0};
   shape20h_(&xi,&et,&ze,xl,&det,shp,&flag);if(!(det>0))fail("nonpositive reference Jacobian");
   for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int n=0;n<20;n++)F[3*i+j]+=ul[3*n+i]*shp[4*n+j];
   for(int i=0;i<3;i++)for(int j=0;j<3;j++)for(int k=0;k<3;k++)c[3*i+j]+=F[3*k+i]*F[3*k+j];
   const int idx[6]={0,4,8,1,2,5};for(int k=0;k<6;k++)if(!closeval((c[idx[k]]-(k<3?1:0))/2,s->strain[k]))fail("stale native strain/geometry");
   if(!s->plastic)continue;
   double B[360];for(int n=0;n<20;n++)for(int d=0;d<3;d++){
    int col=3*n+d;for(int k=0;k<6;k++){int i=idx[k]/3,j=idx[k]%3;B[60*k+col]=(F[3*d+i]*shp[4*n+j]+F[3*d+j]*shp[4*n+i])*(k<3?.5:1);}
   }
   for(int i=0;i<60;i++)for(int j=0;j<60;j++){double value=0;for(int k=0;k<6;k++)for(int l=0;l<6;l++)value+=B[60*k+i]*s->delta[6*k+l]*B[60*l+j];K[60*i+j]+=det*value;}
   count++;
  }
  for(int i=0;i<60;i++)if(eq[i]>0)for(int j=0;j<60;j++)if(eq[j]>0&&K[60*i+j]!=0)add_sm_st_as_(au,ad,jq,ir,&eq[i],&eq[j],&K[60*i+j],&neutral,&neutral,nzs);
 }
 fprintf(stderr,"THESIS_VOLUME_MATERIAL scattered %d plastic IPs\n",count);
}

/* plcopy.f sets slot802 after either >200-point branch. For declared-zero
 * kinematic data, restore that count before native mechmodel can read unused
 * stack storage. Called immediately after native materialdata_me. */
void thesis_material_table_scope_(const int32_t *declared,double *table){
 if(!thesis_material_enabled())return;
 if(*declared!=0)fail("declared kinematic hardening unsupported");
 table[801]=0.5;
}
