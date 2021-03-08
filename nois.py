# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 19:19:08 2020

@author: Louie
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Aug 22 19:19:08 2020

@author: Louie
"""

import numpy as np
import random as rand
import matplotlib.pyplot as plt
import time as time
import matplotlib
#matplotlib.use('GTK3Agg') 


class Ball():


  
    def __init__(self,size,n,max_vel):
        self.n=n
        self.size=size
        self.max_vel=max_vel
        self.pos_x=np.random.uniform(1,size,(n))
        self.pos_y=np.random.uniform(0,size,(n))
        self.vel_x=np.zeros(n)
        self.vel_y=np.zeros(n)
        self.acc_x=np.zeros(n)
        self.acc_y=np.zeros(n)
        self.x_hist=self.pos_x
        self.y_hist=self.pos_y
    
    def update(self):
        self.pos_x=(self.pos_x+self.vel_x)
        self.pos_y=(self.pos_y+self.vel_y)
        self.vel_x=(self.vel_x+self.acc_x)/5 #we divide here to have stronger 
        self.vel_y=(self.vel_y+self.acc_y)/5 #direction without speeding up 
        self.acc_x=0                         #too much
        self.acc_y=0
        
    def apply_force(self,fx,fy):
        self.acc_x=self.acc_x+fx
        self.acc_y=self.acc_y+fy
        
    def speed_check(self):
        """
        scales velocity vector such that it has a magnitude less than the
        max velocity. Direction is kept the same during scaling
        """
        mag=np.zeros((self.n),dtype=bool)
        mags=self.vel_x**2+self.vel_y**2
        mag=mags>self.max_vel**2
        rescaler=mags[mag]/self.max_vel
        self.vel_x[mag]=self.vel_x[mag]/rescaler
        self.vel_y[mag]=self.vel_y[mag]/rescaler
        
    def edge_check(self):
        """ 
        Applies a torus symmetery, i.e particles loop around
        """
        outside=np.zeros((self.n),dtype=bool)
        outside=self.pos_x<0
        self.pos_x[outside]=self.size-1
        outside=self.pos_x>self.size
        self.pos_x[outside]=0
        outside=self.pos_y<0
        self.pos_y[outside]=self.size-1
        outside=self.pos_y>self.size
        self.pos_y[outside]=0
        
    def get_forces(self,vector_x,vector_y):
        index_x=np.floor(self.pos_x).astype(int)
        index_y=np.floor(self.pos_y).astype(int) #floor as index start at 0
        fx=vector_x[index_x,index_y]
        fy=vector_y[index_x,index_y]
        
        return fx,fy
    
    def drive(self,vector_x,vector_y,m):
        count=0
        while count<m:
            fx,fy=self.get_forces(vector_x,vector_y)
            self.apply_force(fx,fy)
            self.update()
            
            self.x_hist=np.vstack([self.x_hist,self.pos_x])
            self.y_hist=np.vstack([self.y_hist,self.pos_y])
            self.edge_check()
            self.speed_check()
            count=count+1
        return self.x_hist,self.y_hist
            



def inner_grid_setup(sx,ex,sy,ey,n): 
    """ 
    sx= start x, ex = end x...
    n= steps inside the inner grid
    
    so returns something like
    
     (0,0)   (0,0.5)   (0,1)
    (0.5,0) (0.5,0.5) (0.5,1)
     (1,0)   (1,0.5)   (1,1)
    """
    inner_grid=np.zeros((n,n,2))
    x=np.linspace(sx,ex,n)
    y=np.linspace(sy,ey,n)
    xx,yy=np.meshgrid(x,y)
    for i in range(n):
        for j in range(n):
            inner_grid[i][j][0]=xx[i][j]
            inner_grid[i][j][1]=yy[i][j]
   
    return inner_grid

def get_inner_prevals(corners,m):
    """ returns the inner values of the grid"""
    #dotting corners and displacement vectors
    val_a=a.dot(corners[0])
    val_b=b.dot(corners[1])
    val_c=c.dot(corners[2])
    val_d=d.dot(corners[3])
    x=np.linspace(0,1,m)
    y=np.linspace(0,1,m)
    xx,yy=np.meshgrid(x,y)
    #intrpolation using fade function
    ab=val_a+(6*xx**5-15*xx**4+10*xx**3)*(val_b-val_a)
    cd=val_c+(6*xx**5-15*xx**4+10*xx**3)*(val_d-val_c)
    val=ab+(6*yy**5-15*yy**4+10*yy**3)*(cd-ab)
    
    return val 


def perlin(n,m):
    """
    n>m (ints)
    """
    
    
    scale=int(n/m)+1
    vectors=np.random.uniform(-1,1,(scale,scale,2)) #grid of random vectors
    vectors[-1,:]=vectors[0,:]
    vectors[-2,:]=vectors[1,:]
    vectors[:,-1]=vectors[:,0]
    vectors[:,-2]=vectors[:,1]
    data=np.ones((n,n))
    global a,b,c,d
    a=inner_grid_setup(0,1,0,-1,m)  #sets up inner grids
    b=inner_grid_setup(-1,0,0,-1,m)
    c=inner_grid_setup(0,1,1,0,m)
    d=inner_grid_setup(-1,0,1,0,m)
    for i in range(scale-1):
        for j in range(scale-1):
            corners=[vectors[i][j],vectors[i][j+1],vectors[i+1][j],vectors[i+1][j+1]]
            heights=get_inner_prevals(corners, m)
            data[i*m:(i+1)*m,j*m:(j+1)*m]=heights
    
    #plot(n,data)
    return data

def plot(n,data,name="test"):
    #n=2**n
    plt.figure()
    fig = plt.gcf()
    fig.set_size_inches(19.2,10.8)
    plt.clf()
    plt.axis('off')
    
    ax = fig.add_subplot(1,1,1)
    plt.axis('off')
    x=np.linspace(0,n,n)
    y=np.linspace(0,n,n)
    plot=ax.pcolormesh(x, y, data,cmap="Greys")

    #plt.savefig("%s"%(name), dpi=300)
    

def fractal(n,m,k):
     """
     n>k (ints)
     super imposes different frequency perlin noise
     
     """
     maximum=2**n
     data=np.zeros((maximum,maximum))
     for i in range(k):
         temp=perlin(maximum,2**(m-i))*(2**(-(m+i)))
         print(np.shape(temp))
         data=data+temp/k
         #plot(n,data/k,i)
     data=data
     # data=data/np.amax(data)
     # data=1-np.abs(data)-data
     plot(maximum,data)
     return data
 
def curl(n,m,k,balls,updates,max_vel=5,wildness=25,x_scale=50,y_scale=100):
    s=time.time()
    pot=fractal(n,m,k)
    #pot=perlin(2**n,2**k)
    size=len(pot[0])
    vector_x,vector_y=vector_field(pot,wildness,x_scale,y_scale)
    x=np.gradient(vector_y,axis=1)
    y=-np.gradient(vector_x,axis=0)
    x,y=dipoles(x, y, 8, 100000)
    balls=Ball(size=size,n=balls,max_vel=max_vel)
    x_hist,y_hist=balls.drive(x, y, updates)
    
    #plot_flow(x_hist, y_hist)
    non_overlap_plot(x_hist, y_hist)
    
    
    print(time.time()-s)

    
def animate_flow(x_hist,y_hist): 
    plt.style.use("default")
   
    plt.axis('off')
    #c_list=["#AD450C","#FF813D","#FA6E23","#00ADA7","#23FAF2","white","black"]
    #c_list=["#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
    #c_list=["#007D32","#6BFFA6","#00FC65","#5B8069","#00CC50","#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
    #c_list=["#7A2418","#FB8C7D","#FA4932","#7A443D","#C23827"]
    c_list=["black"]
    s_list=[5,5,5,5,10,5,2,15,10]
    #s_list=[5]
    
    for i in range(len(x_hist)):
         plt.figure(1,figsize=(19.20,10.80)) 

         plt.style.use("default")
         plt.axis('off')
        # color=rand.choice(c_list)
        # s=rand.choice(s_list)
         count=0
         color=rand.choice(c_list)
         
         while count<len(x_hist[i]):
             s=rand.choice(s_list)
             step=rand.randint(count, len(x_hist[i]))
             temp_x=x_hist[i][count:count+step]
             temp_y=y_hist[i][count:count+step]
             plt.scatter(temp_x,temp_y,s=s,alpha=0.3,color=color,edgecolors="black",markr="_")
             count=count+step
             
         plt.savefig("%s"%(i), dpi=600)   

        
def non_overlap_plot(x,y):
    x=x.T
    y=y.T
    
    #c_list=["white","white","white","#7A2418","#FB8C7D","#FA4932","#7A443D","#C23827"]
    #c_list=["black","white","#FCED0D","#FFF200","white","black","black","black","white","white"]
    c_list=["#007D32","#6BFFA6","#00FC65","#5B8069","#00CC50","#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
    #c_list=["#38C8F5","#54D7FF","#1686A8","#A86D32","#F58C76","#A86759","#EBDC3B","#ED721A"]
    #c_list=["#1972D1","#D13224","#0ED0D1","#D19124","#17D16A","#5AD1B3","#D18B28"]
    #c_list=["#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
    #c_list=["#7A2418"]
    #c_list=["grey","black","white"]
    
    #c_list=["white"]
    #s_list=[30]
    fig=plt.figure()
    fig.set_size_inches(19.2,10.8)
    plt.tick_params(top=False, bottom=False, left=False, right=False,
                labelleft=False, labelbottom=False)
    
    ax = plt.gca()
    ax.spines['bottom'].set_color('white')
    ax.spines['top'].set_color('white') 
    ax.spines['right'].set_color('white')
    ax.spines['left'].set_color('white')
    
    ax.set_aspect(1)
    
    fig.canvas.draw()
    
    x=np.reshape(x,(np.shape(x)[0]*np.shape(x)[1]))
    y=np.reshape(y,(np.shape(y)[0]*np.shape(y)[1]))
    size=(max(x),max(y))
    size=max(size)
    print(size)
    s_list=size/np.array([100,50,200,400,150,170])
    #s_list=size/np.array([190,200,200,400,150,170,100,85,125,1000,80,90,75,])
    #s_list=size/np.array([100,110,90,85,105,115])*1.25
    #s_list=size/np.array([100])
    count=0
    
    
    # a=rand.randint(0,len(x)/5)
    # b=(len(x)/5-a)*rand.random()
    # a=[a,a+5000,a+10000,a+15000,a+20000,a+25000]
    alpha=[0.15,0.2,0.3,0.4]
    
    while count<len(x):
    #while count<4:    
        color=rand.choice(c_list)
        t=rand.choice(s_list)
        ty=t
        step=int(rand.randint(count, len(x))/10)
        # temp_x=x[a[count]:a[count]+int(b)]
        # temp_y=y[a[count]:a[count]+int(b)]
        
        temp_x=x[count:count+step]
        temp_y=y[count:count+step]
        # temp_x=x
        # temp_y=y
        
        temp_x=(np.around(temp_x/t))*t
        temp_y=(np.around(temp_y/ty))*ty

        points=[(temp_x[i],temp_y[i]) for i in range(len(temp_x))]
        points=list(set(points))
       
        temp_x=np.array(points)[:,0]
        temp_y=np.array(points)[:,1]
                                             

        s = ((ax.get_window_extent().width*(t/(0.5*np.pi) ) / (size) * 72./fig.dpi) ** 2) 
        #r=0.5 data units
        ax.scatter(temp_x,temp_y,s=s*2,alpha=rand.choice(alpha)/2,color=color,)
        #ax.patch.set_facecolor('#ED723B')
        #ax.patch.set_facecolor('#D15028')
        ax.patch.set_alpha(0.5)
        #count=count+1
        count=count+step
    

def plot_flow(x_hist,y_hist):
    """
    This is quite inefficient due to plotting each line seperately. This is needed
    if you want matplotlib to apply alpha properly. Will consider changing libraries
    """
    plt.figure()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(19.2,10.8)
    plt.axis('off')
    
    x_hist=x_hist.transpose()   
    y_hist=y_hist.transpose()

    plt.style.use("default")
    #plt.style.use('dark_background')
 
    #c_list=["#AD450C","#FF813D","#FA6E23","#00ADA7","#23FAF2","white","black"]
    #c_list=["#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
    #c_list=["#007D32","#6BFFA6","#00FC65","#5B8069","#00CC50","#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
    #c_list=["#AD450C","#FF813D","#FA6E23","#00ADA7","#23FAF2","white","black"]
    c_list=["#7A2418","#FB8C7D","#FA4932","#7A443D","#C23827"]
    #c_list=["black","white","#FCED0D","#FFF200","white","black","black","black","white","white"]
    #c_list=["#7A2418"]
    #s_list=[1,1,1,1,5,5,5,5,10,5,2,15,25,30,5,10]
    s_list=[15]
    
    for i in range(len(x_hist)):
        # color=rand.choice(c_list)
        # s=rand.choice(s_list)
        count=0
        while count<len(x_hist[i]):
            color=rand.choice(c_list)
            s=rand.choice(s_list)
            step=rand.randint(count, len(x_hist[i]))
            temp_x=x_hist[i][count:count+step]
            temp_y=y_hist[i][count:count+step]
            plt.scatter(temp_x,temp_y,s=s*2.5,alpha=0.2,color=color,)
            count=count+step
        
       #plt.scatter(x_hist[i],y_hist[i],s=s,alpha=0.11,color=color)
    plt.show()    
    name=str(rand.random())
    #plt.savefig("redddfd",dpi=600)
    #plt.savefig('urban.svg', format='svg', dpi=1200)



def dipoles(vector_x,vector_y,num,strength):
    size=np.shape(vector_x)[0]
    x=np.linspace(0,size,size)
    y=np.linspace(0,size,size)
    xx,yy=np.meshgrid(x,y)
    Ex=np.zeros((size,size))
    Ey=np.zeros((size,size))
    for i in range(num):
        px,py=strength*np.random.uniform(-5,5,size=(2,1))
        pos_x,pos_y=np.random.uniform(0,size,size=(2,1))
        rx=xx-pos_x
        ry=yy-pos_y
        r=(rx*rx+ry*ry)**0.5
        rx_norm=rx/r
        
        ry_norm=ry/r

        dot=rx_norm*px+ry_norm*py
        ex=(3*dot*rx_norm-px)/r**2
        ey=(3*dot*ry_norm-py)/r**2
        print(ex)
        Ex=Ex+ex
        Ey=Ey+ey
        
        # plt.figure()
    # ax = plt.gca()
    # plot=ax.pcolormesh(x, y, grid,cmap="Greys")
        
    plot_vectors(Ex,Ey,size)
    return vector_x+Ex,vector_y+Ey    
         
def plot_vectors(vector_x,vector_y,size):
    plt.figure()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(19.2,10.8)
    plt.axis('off')
    plt.style.use('default')
    plt.axis('off')
    
    x=np.linspace(0,size,size+1)
    y=np.linspace(0,size,size+1)
    xx,yy=np.meshgrid(x,y)
    plt.quiver( xx,yy,vector_x,vector_y,color="black",scale=50)
    plt.show()    
        
def rotate(xs,ys,angles):
    new_xs = np.cos(angles) * (xs) - np.sin(angles) * (ys)
    new_ys = np.sin(angles) * (xs) + np.cos(angles) * (ys)
    return new_xs, new_ys    

def vector_field(data,wildness=25,x_scale=50,y_scale=100):
    size=len(data[0])
    vector_y=np.ones((size,size))
    vector_x=np.zeros((size,size))
    angles=2*np.pi*data*wildness
    vector_x,vector_y=rotate(vector_x,vector_y,angles)
    vector_x=vector_x*x_scale
    vector_y=vector_y*y_scale
    #plot_vectors(vector_x,vector_y,size)
    
    return vector_x,vector_y

def plot_line(data,y):
    
    plt.figure()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(19.2,10.8)
    plt.axis('off')
    plt.style.use('dark_background')
    for i in range(len(data)):
        #c_list=["#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
        #c_list=["#7A2418","#FB8C7D","#FA4932","#7A443D","#C23827"]
        c_list=["#007D32","#6BFFA6","#00FC65","#5B8069","#00CC50","#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
        #c_list=["#7A2418","#FB8C7D","#FA4932","#7A443D","#C23827"]
        color=rand.choice(c_list)
        plt.plot(data[i],y,color=color,alpha=0.25)

def plot_circle(data,theta):
    
    plt.figure()
    plt.clf()
    fig = plt.gcf()
    fig.set_size_inches(19.2,10.8)
    plt.axis('off')
    plt.style.use('dark_background')
    for i in range(len(data)):
        x=data[i]*np.cos(theta)
        y=data[i]*np.sin(theta)
        #c_list=["#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
        #c_list=["#7A2418","#FB8C7D","#FA4932","#7A443D","#C23827"]
        #c_list=["#010408","#007D32","#6BFFA6","#00FC65","#5B8069","#00CC50","#04577A","#53C8FB","#07B1FA","#28627A","#068CC7","white"]
        #c_list=["#7A2418","#FB8C7D","#FA4932","#7A443D","#C23827"]
        c_list=["#AD450C","#FF813D","#FA6E23","#00ADA7","#23FAF2","white","black"]
        color=rand.choice(c_list)
        
        plt.plot(x,y,color=color,alpha=0.12)

def run(n,m,k,balls,updates,max_vel=5,wildness=25,x_scale=500,y_scale=1000):
    """ n>k (ints)
    
    """
    s=time.time()
    data=fractal(n,m,k)
    
    size=len(data[0])
    vector_x,vector_y=vector_field(data,wildness,x_scale,y_scale)
    #vector_x,vector_y=dipoles(vector_x, vector_y, 1, 100000)
    
    
    balls=Ball(size=size,n=balls,max_vel=max_vel)
    x_hist,y_hist=balls.drive(vector_x, vector_y, updates)
    
    plot_flow(x_hist, y_hist)
    #non_overlap_plot(x_hist, y_hist)
    #plot(n,data)
    
    print(time.time()-s)
    #animate_flow(x_hist, y_hist)
    
def run_lines(n,k):
    var=fractal(n,k)*9000
    #var=perlin(2**n,2**k)*80
    size=np.shape(var)[0]
    x=np.linspace(0,size,size)
    y=np.linspace(0,size,size)
    X,Y=np.meshgrid(x,y)
    data=var+X
    columns=[]
    for i in range(np.shape(data)[0]):
        temp=data[:,i]
        columns.append(temp)
    plot_line(columns,y)    

def run_circle(n,m,k):    
    #var=perlin(2**n,2**k)*800
    var=fractal(n,m,k)*40000
    size=np.shape(var)[0]
    r=np.linspace(size/9,size,size)
    # b=np.linspace(size/4,size/2,int(size/2))
    # r=np.vstack([r,b])
    theta=np.linspace(0,1.998*np.pi,size)
    R,Theta=np.meshgrid(r,theta)
    data=var+R
    columns=[]
    for i in range(np.shape(data)[0]):
        temp=data[:,i]
        # temp2=temp[::-1]
        # temp=temp+temp2
        columns.append(temp)
    flip=-np.array(columns)*(-1)
    plot_circle(columns,theta)
    # theta2=np.linspace(0,-np.pi,size)
    # plot_circle(columns,theta2)    
#
run(7,5,2,3000,1000,max_vel=5,wildness=50,x_scale=50,y_scale=50)
#curl(7,6,3,3000,1000,max_vel=5,wildness=25,x_scale=50,y_scale=50)
#curl(9,3,5000,1000,max_vel=25,wildness=15,x_scale=5,y_scale=10)
#perlin(2**9,2**3)
#fractal(11,7,4)
#run_lines(10,9)
#run_circle(6,3,2)


