import sys; sys.path.insert(0, r"C:\OpenSim 4.5\sdk\Python")
import numpy as np, opensim as osim
M=r"C:\Users\bubli\Documents\OpenSim\4.5\Models\Gait10dof18musc"
def readmot(f):
    L=open(f).read().splitlines()
    i=[k for k,l in enumerate(L) if l.strip().lower()=='endheader'][0]
    cols=L[i+1].split('\t'); cols=[c.strip() for c in cols if c.strip()]
    d=np.array([[float(x) for x in l.split()] for l in L[i+2:] if l.strip()])
    return cols,d
c,ik=readmot(M+r"\IK\subject01_walk_IK.mot"); t=ik[:,0]
cg,g=readmot(M+r"\subject01_walk_grf.mot")
fyR=g[:,cg.index('ground_force_vy')]; fyL=g[:,cg.index('1_ground_force_vy')]; tg=g[:,0]
for name,fy in (('R',fyR),('L',fyL)):
    on=fy>20; ch=np.flatnonzero(np.diff(on.astype(int)))
    print(name,'contact changes at', [(round(tg[k+1],3), 'ON' if on[k+1] else 'OFF') for k in ch])
print('pelvis_tx range',ik[:,2].min(),ik[:,2].max())
m=osim.Model(M+r"\subject01.osim"); s=m.initSystem()
cs=m.getCoordinateSet()
calR=m.getBodySet().get('calcn_r'); toeR=m.getBodySet().get('toes_r'); talR=m.getBodySet().get('talus_r')
xs=[]
for k in range(len(t)):
    for j,n in enumerate(c[1:]):
        v=ik[k,j+1]; 
        if cs.get(n).getMotionType()==1: v=np.radians(v)
        cs.get(n).setValue(s,v,False)
    m.realizePosition(s)
    xs.append([calR.getPositionInGround(s).get(0),calR.getPositionInGround(s).get(1),toeR.getPositionInGround(s).get(0),toeR.getPositionInGround(s).get(1),talR.getPositionInGround(s).get(1)])
xs=np.array(xs)
v=np.gradient(xs[:,0],t)
for k in range(0,len(t),6): print(round(t[k],3), 'calcR x %.3f y %.3f toe x %.3f y %.3f talus y %.3f  vx %.2f'%(tuple(xs[k])+(v[k],)))
# default pose geometry
for n in c[1:]: cs.get(n).setValue(s,0,False)
cs.get('pelvis_ty').setValue(s,0,False); m.realizePosition(s)
for b in ['pelvis','femur_r','tibia_r','talus_r','calcn_r','toes_r','torso']:
    p=m.getBodySet().get(b).getPositionInGround(s); print('default',b,round(p.get(0),4),round(p.get(1),4),round(p.get(2),4))
