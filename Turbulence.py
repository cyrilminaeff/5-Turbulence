# -*- coding: utf-8 -*-
"""
Created on Sun May 19 22:27:04 2019

@author: cyril
Turbulence
"""

import numpy as np
import matplotlib.pyplot as plt
import random
import math
from math import pi
import time

start_time = time.time()

Lx, Ly = 0.01, 0.01
Nx, Ny, Nt = 64, 64, 40
dx, dy = Lx/Nx, Ly/Ny
A_vozmush = Lx*10
dt = 1e-8

R = 8.314 / 0.02904 # air
G = 1.4
Cv = R / (G-1)
T0 = 300
p0 = 10**5
A_p = 1.0 # относительное давление на скачке на левой стенке
V0 = 0.0
c0 = (G*R*T0)**0.5
mu = 1.2 # коэф линейной вязкости
print('Критерий Куранта = ', c0*dt/dx)

u, u_pr = np.ones((Nx, Ny))*V0, np.zeros((Nx, Ny))
v, v_pr = np.ones((Nx, Ny))*V0, np.zeros((Nx, Ny))
E, E_pr = np.ones((Nx, Ny))*Cv*T0, np.zeros((Nx, Ny))
p = np.ones((Nx, Ny))*p0
p[0,:] = p0 * A_p # скачок давления на левой границе
rho = p / (G-1) / E

kx, ky = np.zeros(int(Nx/2)+1), np.zeros(int(Ny/2)+1)   # массивы волновых чисел    
for n in range(0,int(Nx/2)+1):
    kx[n] = n* 2*pi / Lx
for n in range(0,int(Ny/2)+1):
    ky[n] = n* 2*pi / Ly

#-----------------------------------------------------------------------------
# добавление возмущений
def add_disturbances(A_vozmush, u, v): 
    for i in range(Nx):
        for j in range(Ny):
            g = [random.random(), random.random()]
            mod_v = ( (1-2*g[0])**2 + (1-2*g[1])**2 )**0.5
            u[i,j] = u[i,j] + A_vozmush * (1-2*g[0]) / mod_v
            v[i,j] = v[i,j] + A_vozmush * (1-2*g[1]) / mod_v
    return v, u

#-----------------------------------------------------------------------------
# значения на левом полушаге i-1/2 или j-1/2
def values_on_halfsteps(u, v, p): 
    u_12 = np.zeros((Nx+1, Ny))
    v_12 = np.zeros((Nx, Ny+1))
    px_12 = np.zeros((Nx+1, Ny))
    py_12 = np.zeros((Nx, Ny+1))
    
    # граничные условия: u, v = 0 на стенке
    u_12[0,:], u_12[Nx,:] = 0, 0
    v_12[:,0], v_12[:,Ny] = 0, 0
    px_12[0,:], px_12[Nx,:] = (p0*A_p + p[0,:])/2, (p[Nx-1,:] + p0)/2
    py_12[:,0], py_12[:,Ny] = (p0 + p[:,0])/2, (p[:,Ny-1] + p0)/2
    
    for j in range(Ny):
        for i in range(1,Nx):
                u_12[i,j] = (u[i-1,j] + u[i,j]) / 2
                q = -mu*(u[i,j] - u[i-1,j])/dx # ВСЕГДА РАБОТАЕТ, что мб не правильно
                px_12[i,j] = (p[i-1,j] + p[i,j]) / 2 + q

    for i in range(Nx):
        for j in range(1,Ny):
                v_12[i,j] = (v[i,j-1] + v[i,j]) / 2
                q = -mu*(v[i,j] - v[i,j-1])/dy # ВСЕГДА РАБОТАЕТ, что мб не правильно
                py_12[i,j] = (p[i,j-1] + p[i,j]) / 2 + q
    return u_12, v_12, px_12, py_12

#-----------------------------------------------------------------------------
# промежуточные величины для первого (Эйлерова) этапа
def intermediate_values(u, v, E, p, rho):
    u_12, v_12, px_12, py_12 = values_on_halfsteps(u, v, p) # получение значений на полушаге
    for i in range(Nx):
        for j in range(Ny):    
            u_pr[i,j] = u[i,j] - (px_12[i+1,j]-px_12[i,j])/dx*dt/rho[i,j]
            v_pr[i,j] = v[i,j] - (py_12[i,j+1]-py_12[i,j])/dy*dt/rho[i,j]
            E_pr[i,j] = E[i,j] - ( (px_12[i+1,j]*u_12[i+1,j] - px_12[i,j]*u_12[i,j])/dx + (py_12[i,j+1]*v_12[i,j+1] - py_12[i,j]*v_12[i,j])/dy )*dt/rho[i,j]
    return u_pr, v_pr, E_pr

#-----------------------------------------------------------------------------
# второй (Лагранжев этап) для вычисления переноса массы [стр 62]
def mass_transfer(u_pr, v_pr, rho):
    dMx_12 = np.zeros((Nx+1, Ny))
    dMy_12 = np.zeros((Nx, Ny+1))
    
    # граничные условия: u, v = const на границе
    for j in range(Ny):
        dMx_12[0,j] = 0
        dMx_12[Nx,j] = 0
    for i in range(Nx):
        dMy_12[i,0] = 0
        dMy_12[i,Ny] = 0
    
    for j in range(Ny):
        for i in range(1,Nx):
            if u_pr[i-1,j] + u_pr[i,j] > 0 :
                dMx_12[i,j] = rho[i-1,j]/2 * ( u_pr[i-1,j] + u_pr[i,j] ) *dy*dt
            else:
                dMx_12[i,j] = rho[i,j]/2 * ( u_pr[i-1,j] + u_pr[i,j] ) *dy*dt
                
    for i in range(Nx):
        for j in range(1,Ny):
            if v_pr[i,j-1] + v_pr[i,j] > 0 :
                dMy_12[i,j] = rho[i,j-1]/2 * ( v_pr[i,j-1] + v_pr[i,j] ) *dx*dt
            else:
                dMy_12[i,j] = rho[i,j]/2 * ( v_pr[i,j-1] + v_pr[i,j] ) *dx*dt
    
    return dMx_12, dMy_12 

#-----------------------------------------------------------------------------
# заключительный этап вычисления перераспределения параметров [стр 67]
    
def D(M):
    if M>=0:
        return 1
    else:
        return 0

def final(u, v, E, u_pr, v_pr, E_pr, rho, dMx_12, dMy_12):
    rho_new = np.zeros((Nx, Ny))
    X = np.zeros((3, Nx, Ny))
    X_pr = np.zeros((3, Nx, Ny))
    X_pr[0,:,:], X_pr[1,:,:], X_pr[2,:,:] = u_pr[:,:], v_pr[:,:], E_pr[:,:]
    
    for i in range(Nx):
        for j in range(Ny):
            rho_new[i,j] = rho[i,j] + ( dMx_12[i,j]+dMy_12[i,j] - dMx_12[i+1,j]-dMy_12[i,j+1] ) / (dx*dy)
            for v in range(3): # перебор по u, v, E
                im1, ip1, jm1, jp1 = i-1, i+1, j-1, j+1 
                if i==0:    im1 = 0 # для учета граничных условий
                if i==Nx-1: ip1 = Nx-1
                if j==0:    jm1 = 0
                if j==Ny-1: jp1 = Ny-1
                X[v,i,j] = X_pr[v,im1,j]*dMx_12[i,j]*D(dMx_12[i,j]) + X_pr[v,i,jm1]*dMy_12[i,j]*D(dMy_12[i,j]) + X_pr[v,ip1,j]*dMx_12[i+1,j]*D(-dMx_12[i+1,j]) + X_pr[v,i,jp1]*dMy_12[i,j+1]*D(-dMy_12[i,j+1])
                X[v,i,j] = X[v,i,j] + X_pr[v,i,j] * ( rho[i,j]*dx*dy - (1-D(dMx_12[i,j]))*dMx_12[i,j] - (1-D(dMy_12[i,j]))*dMy_12[i,j] )
                X[v,i,j] = X[v,i,j] + X_pr[v,i,j] * ( - (1-D(-dMx_12[i+1,j]))*dMx_12[i+1,j] - (1-D(-dMy_12[i,j+1]))*dMy_12[i,j+1] )
                X[v,i,j] = X[v,i,j] / ( rho_new[i,j] * dx*dy )
            p[i,j] = (G-1) * rho_new[i,j] * X[2,i,j]            
    return X[0,:,:], X[1,:,:], X[2,:,:], rho_new, p

#-----------------------------------------------------------------------------
# подсчет кинетической энергии
def full_kinetic_energy (u, v, rho):
    E_kin = 0
    for i in range(Nx):
        for j in range(Ny):
            E_kin = E_kin + rho[i,j] * dx*dy * ( (u[i,j])**2 + (v[i,j])**2 )
    return  E_kin      


#-----------------------------------------------------------------------------
# Цикл по временным слоям
E_kin = np.zeros(Nt+1)
E_kin[0] = full_kinetic_energy (u, v, rho)

save = False # нужно ли сохранять картинки
show = False # нужно ли показывать картинки

for n in range(1, Nt+1):
    if n <= 20: u, v = add_disturbances(A_vozmush, u, v) # добавление возмущения
    u_pr, v_pr, E_pr = intermediate_values(u, v, E, p, rho) # первый этап
    dMx_12, dMy_12 = mass_transfer(u_pr, v_pr, rho) # второй этап
    u, v, E, rho, p = final(u, v, E, u_pr, v_pr, E_pr, rho, dMx_12, dMy_12) # заключительный этап
    E_kin[n] = full_kinetic_energy (u, v, rho) # запоминание полной кинетической энергии системы
    
    if n%5 == 0 and (save or show):
        E_kin_arr = np.zeros((Nx, Ny))
        E_kin_arr[:,:] = ( (u[:,:])**2 + (v[:,:])**2 )
        Ef_arr = np.fft.rfft2(E_kin_arr)
        
        plt.clf()
        plt.plot( np.log(kx) , [ np.log(np.abs(Ef_arr[i,i])*Ny) for i in range(len(kx))], label = 'ln E(kx)', marker = 'o' , markersize = 0);
        plt.plot( np.log(kx)  , 16 - 5/3*np.log(kx) , label = '16 - 5/3 ln(k)', marker = 'o' , markersize = 0)
        plt.title('Распределение кин. энергии по волновым числам')
        plt.legend();
        plt.grid()
        plt.ylim(bottom=-2, top=6)
        plt.xlabel('ln k')
        plt.ylabel('ln E(k)')
        if save: plt.savefig('ln_E(k)_n='+str(n)+'.png', fmt='png', dpi=200)
        if show: plt.show()
        
        plt.clf()
        plt.title('ln(E(k))')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(np.log(np.abs(Ef_arr.transpose())), cmap='jet', interpolation='bilinear', origin='lower', vmin=-8 , vmax=2)
        plt.colorbar()
        if save: plt.savefig('ln_Ef_arr_n='+str(n)+'.png', fmt='png', dpi=200)
        if show: plt.show()
        
        mod_v = ( (u)**2 + (v)**2 )**0.5
        plt.clf()
        plt.title('Поле скоростей')
        plt.xlabel('x')
        plt.ylabel('y')
        plt.imshow(mod_v[:,:].transpose(), cmap='jet', interpolation='bilinear', origin='lower', vmin=V0 , vmax=V0+4*A_vozmush)
        plt.colorbar()
        if save: plt.savefig('mod_v_n='+str(n)+'.png', fmt='png', dpi=200)
        if show: plt.show()

# изменение кинетической энергии
plt.plot( [n*dt for n in range(Nt+1)] , [ E_kin[n] for n in range(Nt+1)], label = 'E_kin(t)', marker = 'o' , markersize = 0);
plt.title('Изменение удельной кин. энергии системы во времени')
plt.legend();
plt.grid()
plt.xlabel('t')
plt.ylabel('E_kin(t)')
if save: plt.savefig('E_kin(t).png', fmt='png', dpi=200)
plt.show();

#-----------------------------------------------------------------------------

print("----%s seconds----" % (time.time()-start_time))
