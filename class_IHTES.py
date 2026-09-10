# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 09:15:57 2026

@author: anton
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import image as mpimg
from matplotlib.colors import LogNorm
from scipy import special
from scipy.integrate import simpson,quad
from CoolProp.CoolProp import PropsSI 

"""-----------------------------dry air----------------------------------"""
# temperature Kelvin, pressure Pa
# http://www.coolprop.org/coolprop/HighLevelAPI.html#parameter-table

def cp_a(T,p):
    return PropsSI('C','T',T,'P',p,'Air')  #J/kg/K

def cv_a(T,p):
    return PropsSI('O','T',T,'P',p,'Air') #J/kg/K

def rho_a(T,p):
    return PropsSI('D','T',T,'P',p,'Air') #kg/m3

def mu_a(T,p): #dynamic viscosity
    return PropsSI('V','T',T,'P',p,'Air') #Pa s

def ni_a(T,p): #kinematic viscosity
    return mu_a(T,p)/rho_a(T,p)         #m2/s

def k_a(T,p):
    return PropsSI('L','T',T,'P',p,'Air') #W/K/m



def gamma_a(T,p):    # specific heat ratio
    return cp_a(T,p)/cv_a(T,p)

def he_a(T,p):
    return PropsSI('H','T',T,'P',p,'Air') #J/kg

def T_he_a(H,p): # T from enthalpy
    return PropsSI('T','P',p,'H',H,'Air') #K


mu_0=4*np.pi*1e-7
#coil properties
rho_e_coil=1.68e-8
mu_r_coil=1
mu_coil=mu_0*mu_r_coil
rho_coil=8960
ρCu20=1/58*1e-6  
αCu=0.00393 #ºC-1  # Coeficiente de variación con la temperatura de la resistencia a 20 ºC:

class IHTES:
    def __init__(self,TH_pro,EM_prop,hys_para,geom):
        #-------------- thermal properties material-------------------------------------------

        # kg/m3 - solid density
        # J/kg K - solid thermal capacitance
        # W/m K - solid thermal conductivity
        self.rho_s,self.cp_s,self.k_s=TH_pro
        ep = 0.85  # - solid emissivity

        #PB specifications

        self.e = 0.4  # - void fraction
        dp = 2e-2  # m - solid mean diameter
        self.D_p=dp
        #----------------electromagnetic properties -----------------------------------------
    
        self.sigma_p,self.mu_r_p,self.f_magn,self.rho_magn=EM_prop
        self.rho_e_p=1/self.sigma_p 
        self.mu_p=mu_0*self.mu_r_p
        #----------------------------------------hysteresis----------------------------------------
 
        self.hys_para=hys_para
 

        #-------------------------------electrical resistivity (ohm m) --------------------------


        #---------------relative magnetic permeability---------------------------------------------



        #----------------------magnetic permeability----------------------------------------
   

        
        
        
        #------------------ geometry --------------------------
        D_TES,AR_TES,e_ins,Dt_ou,et,d_coil_r=geom
        self.D_TES=D_TES
        self.e_ins=e_ins #insulaiton
        self.AR_TES=AR_TES
        self.Dt_ou,self.et,self.d_coil_r=Dt_ou,et,d_coil_r #d_coil_r space between windings/Dt_ou
        self.H_TES=D_TES*AR_TES
        self.H_coil=self.H_TES
        self.D_coil=D_TES+2*e_ins
        self.N=self.H_coil/(Dt_ou*(1+d_coil_r))
        self.V_TES=self.H_TES*self.D_TES**2/4*np.pi
        self.M_TES=self.V_TES*(1-self.e)*self.rho_s
        self.C_TES=self.M_TES*self.cp_s
        
   
        self.M_LARCav=np.vectorize(self.M_LARCa)
        self.M_LARCdv=np.vectorize(self.M_LARCd)
        self.A_LARC2v=np.vectorize(self.A_LARC2)

    #-----------------------hysteresis heating model-----------------------------------
    
    def wrevf(self,T):
        t=T-273.15
        return np.interp(t,self.hys_para['x'],self.hys_para['wrev']*1000)
    def wirrf(self,T):
        t=T-273.15
        return np.interp(t,self.hys_para['x'],self.hys_para['wirr']*1000)
    def Hpf(self,T):
        t=T-273.15
        return np.interp(t,self.hys_para['x'],self.hys_para['Hp']*1000)
    def epsrf(self,T):
        t=T-273.15
        return np.interp(t,self.hys_para['x'],self.hys_para['epsr'])
    def Msf(self,T):
        t=T-273.15
        return np.interp(t,self.hys_para['x'],self.hys_para['Ms']*1000)
    
    

    def M_LARCa(self,H,Hm,T):
        epsr,wrev,wirr,Hp,Ms=self.epsrf(T),self.wrevf(T),self.wirrf(T),self.Hpf(T),self.Msf(T)
        c2=2/np.pi*0.5*(np.arctan(2/wirr*(Hm+Hp))-np.arctan(2/wirr*(Hm-Hp)))
        if H == 0:
            Lange=0    
        else:
            Lange=np.cosh(2*H/wrev)/np.sinh(2*H/wrev)-(wrev/(2*H))

        MMs=epsr*Lange+(1-epsr)*(2/np.pi*np.arctan(2/wirr*(H-Hp))+c2)
        return MMs*Ms

    def M_LARCd(self,H,Hm,T):
        epsr,wrev,wirr,Hp,Ms=self.epsrf(T),self.wrevf(T),self.wirrf(T),self.Hpf(T),self.Msf(T)
        c2=2/np.pi*0.5*(np.arctan(2/wirr*(Hm+Hp))-np.arctan(2/wirr*(Hm-Hp)))
        if H == 0:
            Lange=0    
        else:
            Lange=np.cosh(2*H/wrev)/np.sinh(2*H/wrev)-(wrev/(2*H))
        MMs=epsr*Lange+(1-epsr)*(2/np.pi*np.arctan(2/wirr*(H+Hp))-c2)
        return MMs*Ms

    def A_LARC(self,Hm,T):
        epsr,wrev,wirr,Hp,Ms=self.epsrf(T),self.wrevf(T),self.wirrf(T),self.Hpf(T),self.Msf(T)
        logarg=(4*(Hm+Hp)**2+wirr**2)/(4*(Hm-Hp)**2+wirr**2)
        return -Ms*(1-epsr)/np.pi*(wirr*np.log(logarg)-4*Hp*(np.arctan(2*(Hm+Hp)/wirr)+np.arctan(2*(Hm-Hp)/wirr)))




    def A_LARC2(self,Hm,T):
        return (quad(self.M_LARCd,-Hm,+Hm,args=(Hm,T),full_output=0)[0]-quad(self.M_LARCa,-Hm,+Hm,args=(Hm,T),full_output=0)[0])


    def e_hys(self,Hm,T): #W/kgHz #J/kg
        return self.A_LARC(Hm,T)*mu_0/self.rho_magn

    def Phys_sp(self,Hm,T,freq): #W/kg
        
        return self.A_LARC(Hm,T)*freq*mu_0/self.rho_magn*self.f_magn

    def Phys_H(self,Hm,T,freq):
        return self.Phys_sp(Hm,T,freq)*self.V_TES*self.rho_s*(1-self.e)
    
    def Hmf(self,I):
        H=I*self.N/self.H_coil #rms
        return 2**0.5*H
    
    def Hf(self,I): #rms
        H=I*self.N/self.H_coil #rms
        return H
    
    def Phys(self,freq,I,Tmagn):
        Hm=self.Hmf(I)
        
        return self.Phys_sp(Hm,Tmagn,freq)*self.V_TES*self.rho_s*(1-self.e)
    
    #eddis heating model

    #penetration depth on the particle /pebble m
    def depth_p(self,freq):
        return (np.pi*self.mu_p*self.sigma_p*freq)**(-0.5)

   

    # F is the power transmission factor (Fournet, 1985; Orfeuil, 1981), 
    # expressed as a combination of Bessel f unctions of the parameter x 

    def F(self,freq):
        R_p=self.D_p/2
        x=R_p/self.depth_p(freq)*(2**0.5)
        return 2**0.5*(special.ber(x)*special.berp(x)+special.bei(x)*special.beip(x))/(special.ber(x)**2+special.bei(x)**2)


    # equivalent resistance of a granular bed
    def Req(self,freq):
        #correction factor
        Ki=1+0.44*self.D_coil/self.H_coil
        return 2*(1-self.e)*(np.pi*self.D_coil*self.N)**2/(self.H_coil*Ki*self.D_p)*(1e-7*self.mu_r_p*freq/self.sigma_p)**0.5*self.F(freq)


    def Peddy(self,freq,I):
        return self.Req(freq)*I**2  
    
    
    #----------------------------------------------- coil
    def rho_e_coil_T(self,T):
        return ρCu20*(1+αCu*(T-273.15-20))  



    # coil tube electrical resistance
    def R_coil_DC(self,Tcoil):
        Lt=self.D_coil*self.N
        Dt_in=self.Dt_ou-2*self.et #inner tube diameter
        St=(self.Dt_ou**2-Dt_in**2)/4*np.pi #cross section
        return self.rho_e_coil_T(Tcoil)*Lt/St

    def depth_coil(self,freq,Tcoil):
        return (self.rho_e_coil_T(Tcoil)/(np.pi*freq*mu_coil))**0.5

    def R_coil_AC(self,freq,Tcoil):
        eps=(self.N*(np.pi)**0.5*self.Dt_ou/2/self.H_coil)**0.5*(np.pi)**0.5*self.Dt_ou/2/self.depth_coil(freq,Tcoil)
        return self.R_coil_DC(Tcoil)*eps*((np.sinh(2*eps)+np.sin(2*eps))/(np.cosh(2*eps)-np.cos(2*eps)))

    def Pcoil(self,freq,I,Tcoil):
        return self.R_coil_AC(freq,Tcoil)*I**2
    
    
    #--------power electronics------------
    def eta_conv(self, freq):
        """
        Calcula pérdidas por conmutación y eficiencia del conversor usando t_switch interno.

        Parámetros:
        - P_in : potencia de entrada al conversor [W]
        - freq : frecuencia de conmutación [Hz]

        Devuelve:

        - eta_switch : eficiencia del conversor [-]
        """
        t_switch_MOSFET = 250e-9  # 250 ns
        t_switch_IGBT   = 800e-9  # 800 ns
        
        t_switch=t_switch_MOSFET 
        
        eta_switch = 1 - 0.5 * t_switch * freq
        
        return eta_switch
    
    


    def Pamb(self, T_PB, T_amb):
        """
        
        Calcula el coeficiente global de pérdidas térmicas U_L [W/m²·K]
        usando la geometría del tanque, aislamiento de lana de roca y convección externa.
        
        Parámetros:
        - T     : temperatura del tanque o del aislamiento [K]
        - e_ves : espesor del contenedor [m] (opcional, default 1 cm)
        - h_ext : coeficiente de convección externa [W/m²·K] (default 5)
        

        
        Calcula las pérdidas térmicas hacia el ambiente [W]

        Parámetros:
        - T_PB : temperatura del TES [K]
        - T_amb: temperatura ambiente [K]
        - e_ves : espesor del contenedor [m] (opcional, default 1 cm)
        - h_ext : coeficiente de convección externa [W/m²·K] (default 5)

        Devuelve:
        - P_amb: pérdidas térmicas al ambiente [W]
        """
                # Conductividad de la lana de roca dependiente de T
        k_ins = 0.035 + 1e-4 * (T_PB - 300)  # W/m·K
        h_ext=10
        
        # Diámetro efectivo incluyendo contenedor y aislamiento
        D_eff = self.D_TES + 2*(self.e_ins)
        
        # Resistencia por conducción radial
        R_cond = np.log(D_eff / self.D_TES) / (2 * np.pi * k_ins * self.H_TES)
        
        # Resistencia por convección externa
        R_conv = 1 / (h_ext * np.pi * D_eff * self.H_TES)
        
        # Resistencia total
        R_tot = R_cond + R_conv
    
        # Pérdidas al ambiente
        P_amb = (T_PB - T_amb) / R_tot
        return P_amb
    
    
    
    #----------out-----------
    
    def Pind(self,freq,I,T_TES): #induced power
        return self.Phys(freq,I,T_TES)+self.Peddy(freq,I)
    
    def Pch(self,freq,I,T_TES,T_amb):
        return self.Pind(freq,I,T_TES)-self.Pamb(T_TES, T_amb)
    
    def Pel(self,freq,I,Tcoil,T_TES): #electric power to the coil
        return self.Pind(freq,I,T_TES)+self.Pcoil(freq,I,Tcoil)
    
    def Pin(self,freq,I,Tcoil,T_TES): #inlet power to power electronics
        return self.Pel(freq,I,Tcoil,T_TES)/self.eta_conv(freq)
    
    def Pconv(self,freq,I,Tcoil,T_TES): 
        return self.Pin(freq,I,Tcoil,T_TES)-self.Pel(freq,I,Tcoil,T_TES) 
    
    def eta_ind(self,freq,I,Tcoil,T_TES): #induction efficieny
        return self.Pind(freq,I,T_TES)/(self.Pind(freq,I,T_TES)+self.Pcoil(freq,I,Tcoil))
    
    def eta_th(self,freq,I,T_TES,T_amb):
        return self.Pch(freq,I,T_TES,T_amb)/self.Pind(freq,I,T_TES)

    def eta_eddy(self,freq,Tcoil): #only eddy
        return self.Req(freq)/(self.Req(freq)+self.R_coil_AC(freq,Tcoil))
    
    def qv(self,freq,I,T_TES):
        return self.Pind(freq,I,T_TES)/self.V_TES
    
    def qv_eddy(self,freq,I):
        return self.Peddy(freq,I)/self.V_TES
    
    def qv_hys(self,freq,I,T_TES):
        return self.Phys(freq,I,T_TES)/self.V_TES
    
    def hys_eddy(self,freq,I,T_TES):
        return self.qv_hys(freq,I,T_TES)/self.qv_eddy(freq,I)

        
    #function for hysteresis from e_hys as input
    
    def eta_hys_eV(self, eV_hys,freq,I,Tcoil):
        P_hys=eV_hys*freq*self.V_TES*(1-self.e)
        return P_hys/(P_hys+self.Pcoil(freq,I,Tcoil))
    
    def qv_hys_eV(self,eV_hys,freq):
        return eV_hys*freq*(1-self.e)



    #-------------------------------overall-----------------------------    
    
    def operate(self, freq, I, Tcoil, T_TES,T_amb):
        """
        Retorna un diccionario con todas las potencias y eficiencias definidas en la clase
        """
        # Potencias
        P_hys = self.Phys(freq, I, T_TES)
        P_eddy = self.Peddy(freq, I)
        P_ind = self.Pind(freq, I, T_TES)
        P_coil = self.Pcoil(freq, I, Tcoil)
        P_el = self.Pel(freq, I, Tcoil, T_TES)
        P_in = self.Pin(freq, I, Tcoil, T_TES)
        P_ch=self.Pch(freq, I,  T_TES,T_amb)
        P_amb=self.Pamb(T_TES, T_amb)
    
        # Eficiencias
        eta_ind = self.eta_ind(freq, I, Tcoil, T_TES)
        eta_conv = self.eta_conv(freq)
        eta_th=self.eta_th(freq,I,T_TES,T_amb)
        # Densidades volumétricas
        qv_total = self.qv(freq, I, T_TES)
        qv_eddy = self.qv_eddy(freq, I)
        qv_hys = self.qv_hys(freq, I, T_TES)
        hys_eddy = self.hys_eddy(freq, I, T_TES)
    
        return {
            'P_hys': P_hys,
            'P_eddy': P_eddy,
            'P_ind': P_ind,
            'P_coil': P_coil,
            'P_el': P_el,
            'P_in': P_in,
            'P_ch':P_ch,
            'P_amb':P_amb,
            'eta_ind': eta_ind,
            'eta_conv': eta_conv,
            'eta_th': eta_th,
            'qv_total': qv_total,
            'qv_eddy': qv_eddy,
            'qv_hys': qv_hys,
            'hys_eddy': hys_eddy
        }

    def charging(self, freq, I, Tcoil, T_TES_0, T_amb, ch_time, timestep):
    
        n_steps = int(ch_time // timestep)
    
        time = np.arange(0, ch_time + timestep, timestep)
        T_TES = np.zeros(n_steps + 1)
        T_TES[0] = T_TES_0
    
        history = []
    
        # Energías acumuladas [J]
        E_ch = 0.0
        E_amb = 0.0
        E_coil = 0.0
        E_in = 0.0
    
        for i in range(n_steps):
    
            results = self.operate(freq, I, Tcoil, T_TES[i], T_amb)
    
            history.append(results.copy())
    
            P_ch   = results['P_ch']
            P_amb  = results['P_amb']
            P_coil = results['P_coil']
            P_in   = results['P_in']
    
            # ---- Balance térmico ----
            dT = (P_ch * timestep) / (self.M_TES * self.cp_s)
            T_TES[i+1] = T_TES[i] + dT
    
            # ---- Energías acumuladas ----
            E_ch   += P_ch   * timestep
            E_amb  += P_amb  * timestep
            E_coil += P_coil * timestep
            E_in   += P_in   * timestep
    
        return time, T_TES, history, E_ch, E_amb, E_coil, E_in