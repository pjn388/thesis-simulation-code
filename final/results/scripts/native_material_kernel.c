/* Isolated fixed-history isotropic material derivative. No solver linkage yet.
 * Inputs are read-only row-major tensors and native returned PEEQ. Output uses
 * stress [11,22,33,12,13,23], engineering Green strain columns. */
#include <math.h>
#include <string.h>
static void mul(const double *a,const double *b,double *c){
  for(int i=0;i<3;i++)for(int j=0;j<3;j++){
    c[3*i+j]=0;for(int k=0;k<3;k++)c[3*i+j]+=a[3*i+k]*b[3*k+j];
  }
}
static double tr(const double *a){return a[0]+a[4]+a[8];}
static double dot(const double *a,const double *b){double s=0;for(int i=0;i<9;i++)s+=a[i]*b[i];return s;}
/* Disabled branch requires no valid constitutive pointers and changes no state. */
int thesis_material_delta(int enabled,const double *c,const double *m,
 double q,double qp,double mu,double bulk,int nt,const double *ep,
 const double *ys,double *out){
  memset(out,0,36*sizeof(double));if(!enabled)return 0;
  if(nt<2 || mu<=0 || bulk<=0 || qp<q-1e-12)return 1;
  for(int i=0;i<9;i++)if(!isfinite(c[i])||!isfinite(m[i]))return 2;
  for(int i=0;i<nt;i++)if(!isfinite(ep[i])||!isfinite(ys[i]) ||
    (i && (ep[i]<=ep[i-1] || ys[i]<ys[i-1])))return 3;
  double det=c[0]*(c[4]*c[8]-c[5]*c[7])-c[1]*(c[3]*c[8]-c[5]*c[6])+c[2]*(c[3]*c[7]-c[4]*c[6]);
  if(!(det>0))return 4;
  double ci[9]={c[4]*c[8]-c[5]*c[7],c[2]*c[7]-c[1]*c[8],c[1]*c[5]-c[2]*c[4],
   c[5]*c[6]-c[3]*c[8],c[0]*c[8]-c[2]*c[6],c[2]*c[3]-c[0]*c[5],
   c[3]*c[7]-c[4]*c[6],c[1]*c[6]-c[0]*c[7],c[0]*c[4]-c[1]*c[3]};
  for(int i=0;i<9;i++)ci[i]/=det;
  double j=sqrt(det),z=pow(j,2./3.),t=dot(c,m)/(3*z),trial[9],sc[9],sc2[9];
  for(int i=0;i<9;i++)trial[i]=mu*(m[i]/z-t*ci[i]);
  mul(trial,c,sc);mul(sc,sc,sc2);double norm=sqrt(fmax(0,tr(sc2)));
  double a=sqrt(2./3.),cop=(qp-q)/a,y=ys[0],h=0;
  if(qp>=ep[nt-1])y=ys[nt-1];
  else if(qp>=ep[0])for(int i=1;i<nt;i++)if(qp<ep[i]){
    h=(ys[i]-ys[i-1])/(ep[i]-ep[i-1]);y=ys[i-1]+h*(qp-ep[i-1]);break;
  }
  int active=cop>0;if(active && (!(norm>0)||!(t>0)))return 5;
  double exact[36],frozen[36];int idx[6]={0,4,8,1,2,5};
  for(int col=0;col<6;col++){
    double dc[9]={0},tmp[9],dci[9],dtrial[9],dsc[9],x[9],v[9];
    dc[idx[col]]=col<3?2:1;if(col>=3)dc[(idx[col]%3)*3+idx[col]/3]=1;
    double dj=j*dot(ci,dc)/2,dz=2*z*dj/(3*j),dt=dot(dc,m)/(3*z)-t*dz/z;
    mul(ci,dc,tmp);mul(tmp,ci,dci);
    for(int i=0;i<9;i++){dci[i]=-dci[i];dtrial[i]=mu*(-m[i]*dz/(z*z)-dt*ci[i]-t*dci[i]);}
    mul(dtrial,c,dsc);mul(trial,dc,x);for(int i=0;i<9;i++)dsc[i]+=x[i];
    mul(sc,dsc,v);double dn=norm>0?tr(v)/norm:0;
    for(int mode=0;mode<2;mode++){
      double f=0,df=0;
      if(active){
        double dcop=(dn-a*y*(mode?0:dj)-2*mu*cop*dt)/(a*a*j*h+2*mu*t);
        f=2*mu*t*cop/norm;df=2*mu*((dt*cop+t*dcop)/norm-t*cop*dn/(norm*norm));
      }
      for(int row=0;row<6;row++){
        int k=idx[row];double value=bulk*j*dj*ci[k]+bulk/2*(det-1)*dci[k]+(1-f)*dtrial[k]-df*trial[k];
        (mode?frozen:exact)[6*row+col]=value;
      }
    }
  }
  for(int i=0;i<6;i++)for(int k=0;k<6;k++){
    out[6*i+k]=exact[6*i+k]-(frozen[6*i+k]+frozen[6*k+i])/2;
    if(!isfinite(out[6*i+k]))return 6;
  }
  return 0;
}
