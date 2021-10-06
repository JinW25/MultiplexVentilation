import numpy as py
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from pylab import show
from scipy.integrate import odeint
from plotly.subplots import make_subplots

# This function is used to simulate the estimated graph for the current patient in ramp waveform
class VCRampModel:
    def __init__(self, Weight, E, R, RR, PEEP, VT, I_E):
        # Defining variables for patient
        self.Weight = Weight
        self.E = E
        self.R = R
        self.RR = RR            # Respiratory rate (frequency, times/min)
        self.I_E = I_E          # IE ratio
        self.PEEP = PEEP        # PEEP pressure
        self.VT = VT            # Tidal Volume

    def simulate(self):
        # Defining settings for ventilator
        Weight = self.Weight
        E = self.E
        R = self.R
        RR = self.RR
        I_E = self.I_E
        PEEP = self.PEEP
        VT = (self.VT)/1000
        timeconst = R/E    # Time constant for exponential curve

        #-----------------KEPT CONSTANT---------------------------------------
        Rt = 0.1      # Fraction for rise time during inspiration
        Dt = 0.999    # Fraction for exponential curve during expiration
        Mt = 1/10     # Time required to reach minimum flow rate
        #----------------------------------------------------------------------

        # Defining parameters
        T = 1/(RR/60)  # Respiratory Rate (Period)
        ti = T*(1-1/(1+I_E))
        te = T-ti                 # Expiration period
        tr = Rt*ti                # Rise time for flow rate during inspiration
        tee = Dt*te               # Time required to complete exponential curve
        tm = ti + tee*Mt          # Time for dropping exponential curve
        tep = tm + (1-Mt)*tee     # Time for rising exponential curve
        dt = 0.001                # Step size for time

        
        # Defining number of period
        Ti = 5

        #-----------------------------------------Simulating the ventilation data for one cycle---------------------------------------------
        # Defining time period for each equation
        t1 = py.arange(0, tr + dt, dt)
        t2 = py.arange(t1[-1], ti + dt, dt)
        t3 = py.arange(t2[-1], tm + dt , dt)
        t4 = py.arange(t3[-1], tep + dt, dt)
        t5 = py.arange(t4[-1], T , dt)

        Qin = 2*VT/(t2[-1]-t1[0]) # Calculating peak flow based on tidal volume and I:E ratio
  
        #------------------------------Defining function handle for each equations for flow rate and volume---------------------------------

        A = Qin/(t1[-1] - t1[0])         # Constant for Q1 and V1
        B = -Qin/(t2[-1]-t2[0])          # Constant for Q2 and V2
        
        # Defining function handle for each equations for volume after integration
        C1 = -(A/2)*(t1[0]-t1[-1])**2-Qin*t1[0]  # Constant for V1
        V1 = lambda t: (A/2)*(t-t1[-1])**2 + Qin*t + C1
        C2 = V1(t1[-1]) - (B/2)*(t2[0]-t2[-1])**2            # Constant for V2
        V2 = lambda t: B/2*(t-t2[-1])**2 + C2

        Q1 = lambda t: A*(t-t1[-1]) + Qin  
        Q2 = lambda t: B*(t-t2[-1])

        # Defining Qout to maintain a constant PEEP for pressure during the expiration period
        # For P4 equals to zero
        t_func = lambda t: R*(1-py.exp(-(t3[-1]-t2[-1])/timeconst))*py.exp(-(t-t3[-1])/timeconst)-E*timeconst*py.exp(-(t-t3[-1])/timeconst)*(1-py.exp(-(t3[-1]-t2[-1])/timeconst))+E*timeconst*(1-py.exp(-(t3[-1]-t2[-1])/timeconst))+E*(t3[-1]+timeconst*py.exp(-(t3[-1]-t2[-1])/timeconst))-E*(t2[-1]+timeconst)
        Qout = -E*V2(t2[-1])/t_func(t4[0])
        
        A3 = Qout                                          # Constant for Q3
        Q3 = lambda t: A3*(1-py.exp(-(t-t3[0])/timeconst))
        A4 = Q3(t3[-1])                                    # Constant for Q4
        Q4 = lambda t: A4*py.exp(-(t-t4[0])/timeconst)
        A5 = Q4(t4[-1])/(t4[-1]-t5[-1])**2
        Q5 = lambda t: A5*(t-t5[-1])**2
        C3 = V2(t2[-1]) -A3*(t2[-1] + timeconst)        # Constant for V3
        V3 = lambda t: A3*(t + timeconst*py.exp(-(t-t2[-1])/timeconst)) + C3
        C4 = V3(t3[-1]) + timeconst*A4
        V4 = lambda t: -A4*timeconst*py.exp(-(t-t3[-1])/timeconst) + C4
        C5 = V4(t4[-1]) -(A5/3)*(t4[-1]-t5[-1])**3
        V5 = lambda t: (A5/3)*(t-t5[-1])**3 + C5
        #------------------------------------------------------------------------------------------------------------------------------

        # By implementing linear single compartment equation to calculate pressure
        P1 = E*V1(t1) + R*Q1(t1) + PEEP
        P2 = E*V2(t2) + R*Q2(t2) + PEEP
        P3 = E*V3(t3) + R*Q3(t3) + PEEP
        P4 = E*V4(t4) + R*Q4(t4) + PEEP
        P5 = E*V5(t5) + R*Q5(t5) + PEEP

        # Storing value for t,P,Q, and V
        t = py.concatenate([t1,t2,t3,t4,t5])
        Q = py.concatenate([Q1(t1),Q2(t2),Q3(t3),Q4(t4),Q5(t5)])
        V = py.concatenate([V1(t1),V2(t2),V3(t3),V4(t4),V5(t5)])
        P = py.concatenate([P1,P2,P3,P4,P5])
        print(Q5(t5))

        # Creating several periods
        for i in range(1,Ti):

            # Creating time for different periods
            t1 = py.arange(i*T, tr + T*i + dt, dt)
            t2 = py.arange(t1[-1], ti + T*i + dt, dt)
            t3 = py.arange(t2[-1], tm + T*i + dt , dt)
            t4 = py.arange(t3[-1], tep + T*i + dt, dt)
            t5 = py.arange(t4[-1], T + T*i, dt)

            # Generating values for different period
            t_temp = py.concatenate([t1,t2,t3,t4,t5])
            t = py.concatenate([t,t_temp])
            Q_temp = Q
            Q = py.concatenate([Q,Q_temp])
            V_temp = V
            V = py.concatenate([V,V_temp])
            P_temp = P
            P = py.concatenate([P,P_temp])
        PIP = round(max(P),2)

        VT_mass = round(VT*1000/Weight,2)
        Qin = round(Qin*60,2) # Converting to the unit of L/min
            
        # Creating subplots using plotly
        figVR1 = make_subplots(rows=4, cols=1,  
                                    vertical_spacing=0.1,
                                    specs=[[{"type": "scatter"}],
                                        [{"type": "scatter"}],
                                        [{"type": "scatter"}],
                                        [{"type": "table"}],
                                        ])


        # Plotting the graph using plotly for pressure, flow, and volume
        figVR1.add_trace(go.Scatter(x=t,y=P, name="Pressure"), row=1, col=1 )
        figVR1.add_trace(go.Scatter(x=t,y=Q, name="Flow"), row=2, col=1 )
        figVR1.add_trace(go.Scatter(x=t,y=V, name="Volume"), row=3, col=1 )

        # Summarise on the MV setting 
        figVR1.add_trace(go.Table(header=dict(values=['PIP (cmH2O)', 'PEEP (cmH2O)', 'RR (1/min)', 'I:E', 'VT (mL/kg)', 'Qin (L/min)'],
                                            font=dict(size=16)
                                            ),
                                    cells=dict(values=[[PIP],[PEEP],[RR],[I_E],[VT_mass],[Qin]],
                                            font=dict(size=16),
                                            height=30
                                            ),
                                            ),
                                            row=4, col=1,                             
                                            )

        # Update xaxis properties
        figVR1.update_xaxes(title_text="<b>Time(s)<b>", title_font=dict(size=18), row=3, col=1)

        # Update yaxis properties
        figVR1.update_yaxes(title_text="<b>Pressure (cmH2O)<b>", title_font=dict(size=15), row=1, col=1) 
        figVR1.update_yaxes(title_text="<b>Flow (L/s)<b>", title_font=dict(size=15),  row=2, col=1) 
        figVR1.update_yaxes(title_text="<b>Volume (L)<b>", title_font=dict(size=15),  row=3, col=1) 
        # Updating figure layout
        figVR1.update_layout(showlegend=False, title_text="Estimated Ventilator Graph for Current Patient", 
                            title_font=dict(size=25),
                            height=800
                            )
        divVR1 = figVR1.to_html(full_html=False)
        return divVR1
        


# This function is used to simulate the estimated graph for the current patient in square waveform
class VCSquareModel:
    def __init__(self, Weight, E, R, RR, PEEP, VT, I_E):
        # Defining variables for patient
        self.Weight = Weight
        self.E = E
        self.R = R
        self.RR = RR            # Respiratory rate (frequency, times/min)
        self.I_E = I_E          # IE ratio
        self.PEEP = PEEP        # PEEP pressure
        self.VT = VT            # Peak Pressure

    def simulate(self):
        # Defining settings for ventilator
        Weight = self.Weight
        E = self.E
        R = self.R
        RR = self.RR
        I_E = self.I_E
        PEEP = self.PEEP
        VT = (self.VT)/1000
        timeconst = R/E    # Time constant for exponential curve
        #----------------------KEPT CONSTANT---------------------------------------
        Rt = 1/10      # Fraction for rise time during inspiration
        Rd = 1/10      # Fraction for drop time during inspiration
        Rp = 1-Rt-Rd   # Fraction for pausing time during inspiration
        Dt = 0.999     # Fraction for exponential curve during expiration
        Mt = 1/3       # Time required to reach minimum flow rate
        #--------------------------------------------------------------------------

        # Defining parameters
        T = 1/(RR/60)  # Respiratory Rate (Period)
        ti = T*(1-1/(1+I_E))
        Qin = VT/(ti*(0.5*Rp + 0.5))     # Peak flow (To be calculated based on the input I:E and Tidal Volume)
        te = T-ti                 # Expiration period
        tr = Rt*ti                # Rise time for flow rate during inspiration
        tp = tr + ti*Rp           # End of pausing time
        tee = Dt*te               # Time required to complete exponential curve
        tm = ti + tee*Mt          # Time for dropping exponential curve
        tep = tm + (1-Mt)*tee     # Time for rising exponential curve
        dt = 0.001                # Step size for time

        
        # Defining number of period
        Ti = 5

        # Defining time period for each equation
        t1 = py.arange(0, tr + dt, dt)
        t2 = py.arange(t1[-1], tp + dt, dt)
        t3 = py.arange(t2[-1], ti + dt , dt)
        t4 = py.arange(t3[-1], tm + dt, dt)
        t5 = py.arange(t4[-1], tep + dt, dt)
        t6 = py.arange(t5[-1], T, dt)

        #------------------------------Defining function handle for each equations for flow rate and volume---------------------------------
        # Defining function handle for each equations for flow rate
        A1 = Qin/(t1[-1] - t1[0])   # Constant for Q1
        Q1 = lambda t: A1*(t-t1[0])  
        Q2 = lambda t: Qin
        A3 = -Qin/(t3[-1] - t3[0])
        Q3 = lambda t: A3*(t-t3[0]) + Qin
        
        # Defining function handle for each equations for volume after integration
        C1 = 0  # Constant for V1
        V1 = lambda t: A1*(t-t1[0])**2/2 
        C2 = V1(t1[-1]) - Qin * t1[-1]              # Constant for V2
        V2 = lambda t: Qin*t + C2
        C3 = V2(t2[-1]) + A3*(t2[-1]**2/2) - Qin*t2[-1] # Constant for V3
        V3 = lambda t: A3*((t**2)/2-t*t3[0]) + Qin*t + C3

        # Defining Qout to maintain a constant PEEP during expiration
        # For P4 equals to zero
        t_func = lambda t: R*(1-py.exp(-(t4[-1]-t3[-1])/timeconst))*py.exp(-(t-t4[-1])/timeconst)-E*timeconst*py.exp(-(t-t4[-1])/timeconst)*(1-py.exp(-(t4[-1]-t3[-1])/timeconst))+E*timeconst*(1-py.exp(-(t4[-1]-t3[-1])/timeconst))+E*(t4[-1]+timeconst*py.exp(-(t4[-1]-t3[-1])/timeconst))-E*(t3[-1]+timeconst)
        Qout = -E*V3(t3[-1])/t_func(t5[0])
        
        A4 = Qout                                          # Constant for Q3
        Q4 = lambda t: A4*(1-py.exp(-(t-t4[0])/timeconst))
        A5 = Q4(t4[-1])                                    # Constant for Q4
        Q5 = lambda t: A5*py.exp(-(t-t5[0])/timeconst)
        A6 = Q5(t5[-1])/(t5[-1]-t6[-1])**2
        Q6 = lambda t: A6*(t-t6[-1])**2
        C4 = V3(t3[-1]) -A4*(t3[-1] + timeconst)        # Constant for V3
        V4 = lambda t: A4*(t + timeconst*py.exp(-(t-t3[-1])/timeconst)) + C4
        C5 = V4(t4[-1]) + timeconst*A5
        V5 = lambda t: -A5*timeconst*py.exp(-(t-t4[-1])/timeconst) + C5
        C6 = V5(t5[-1]) -(A6/3)*(t5[-1]-t6[-1])**3
        V6 = lambda t: (A6/3)*(t-t6[-1])**3 + C6
        #-----------------------------------------------------------------------------------------------------------------------------------

        # By implementing linear single compartment equation to calculate pressure
        P1 = E*V1(t1) + R*Q1(t1) + PEEP
        P2 = E*V2(t2) + R*Q2(t2) + PEEP
        P3 = E*V3(t3) + R*Q3(t3) + PEEP
        P4 = E*V4(t4) + R*Q4(t4) + PEEP
        P5 = E*V5(t5) + R*Q5(t5) + PEEP
        P6 = E*V6(t6) + R*Q6(t6) + PEEP

        # Storing value for t,P,Q, and V
        t = py.concatenate([t1,t2,t3,t4,t5,t6])
        Q = py.concatenate([Q1(t1),Q2(t2)*py.ones(len(t2)),Q3(t3),Q4(t4),Q5(t5),Q6(t6)])
        V = py.concatenate([V1(t1),V2(t2)*py.ones(len(t2)),V3(t3),V4(t4),V5(t5),V6(t6)])
        P = py.concatenate([P1,P2,P3,P4,P5,P6])
            
        for i in range(1,Ti):
            # Defining time period for each equation
            t1 = py.arange(i*T, tr + T*i + dt, dt)
            t2 = py.arange(t1[-1], tp + T*i + dt, dt)
            t3 = py.arange(t2[-1], ti + T*i + dt , dt)
            t4 = py.arange(t3[-1], tm + T*i + dt, dt)
            t5 = py.arange(t4[-1], tep + T*i + dt, dt)
            t6 = py.arange(t5[-1], T + T*i, dt)

            t_temp = py.concatenate([t1,t2,t3,t4,t5,t6])
            t = py.concatenate([t,t_temp])
            Q_temp = Q
            Q = py.concatenate([Q,Q_temp])
            V_temp = V
            V = py.concatenate([V,V_temp])
            P_temp = P
            P = py.concatenate([P,P_temp])

        PIP = round(max(P),2)
        VT_mass = round(VT*1000/Weight,2)
        Qin = round(Qin*60,2) # Converting peak flow to unit of L/min

        # Creating subplots using plotly
        figVR1 = make_subplots(rows=4, cols=1,  
                                    vertical_spacing=0.1,
                                    specs=[[{"type": "scatter"}],
                                        [{"type": "scatter"}],
                                        [{"type": "scatter"}],
                                        [{"type": "table"}],
                                        ])


        # Plotting the graph using plotly for pressure, flow, and volume
        figVR1.add_trace(go.Scatter(x=t,y=P, name="Pressure"), row=1, col=1 )
        figVR1.add_trace(go.Scatter(x=t,y=Q, name="Flow"), row=2, col=1 )
        figVR1.add_trace(go.Scatter(x=t,y=V, name="Volume"), row=3, col=1 )

        # Summarise on the MV setting 
        figVR1.add_trace(go.Table(header=dict(values=['PIP (cmH2O)', 'PEEP (cmH2O)', 'RR (1/min)', 'I:E', 'VT (mL/kg)', 'Qin (L/min)'],
                                            font=dict(size=16)
                                            ),
                                    cells=dict(values=[[PIP],[PEEP],[RR],[I_E],[VT_mass],[Qin]],
                                            font=dict(size=16),
                                            height=30
                                            ),
                                            ),
                                            row=4, col=1,                             
                                            )

        # Update xaxis properties
        figVR1.update_xaxes(title_text="<b>Time(s)<b>", title_font=dict(size=18), row=3, col=1)

        figVR1.update_yaxes(title_text="<b>Pressure (cmH2O)<b>", title_font=dict(size=15), row=1, col=1) 
        figVR1.update_yaxes(title_text="<b>Flow (L/s)<b>", title_font=dict(size=15),  row=2, col=1) 
        figVR1.update_yaxes(title_text="<b>Volume (L)<b>", title_font=dict(size=15),  row=3, col=1) 

        # Updating figure layout
        figVR1.update_layout(showlegend=False, title_text="Estimated Ventilator Graph for Current Patient", 
                            title_font=dict(size=25),
                            height=800
                            )
        divVR1 = figVR1.to_html(full_html=False)
        return divVR1
        


# This function is used to simulate the contour plots in ramp waveform
class VCRampPairing:
    def __init__(self, RR, PEEP, VT, I_E):
        # Defining variables for patient
        self.RR = RR            # Respiratory rate (frequency, times/min)
        self.I_E = I_E          # IE ratio
        self.PEEP = PEEP        # PEEP pressure
        self.VT = VT            # Peak Pressure
    
    def simulate(self):
        # Range of Elastance and Resistance to be Simulated (Any Values larger than 0, with interval 0.5)
        E_range = py.arange(1,52,5)
        R_range = py.arange(1,52,5)
        RR = self.RR
        I_E = self.I_E
        PEEP = self.PEEP
        VT = (self.VT)/1000
        #-----------------KEPT CONSTANT---------------------------------------
        Rt = 0.1     # Fraction for rise time during inspiration
        Dt = 0.999    # Fraction for exponential curve during expiration
        Mt = 1/10     # Time required to reach minimum flow rate
        #----------------------------------------------------------------------
        # Defining parameters
        T = 1/(RR/60)  # Respiratory Rate (Period)
        ti = T*(1-1/(1+I_E))
        te = T-ti                 # Expiration period
        tr = Rt*ti                # Rise time for flow rate during inspiration
        tee = Dt*te               # Time required to complete exponential curve
        tm = ti + tee*Mt          # Time for dropping exponential curve
        tep = tm + (1-Mt)*tee     # Time for rising exponential curve
        dt = 0.001                # Step size for time
        # Creating empty variable for data storage
        T_data = []
        P_suc = []
        E_suc = []
        R_suc = []

        for j in range(len(R_range)):
            for i in range(len(E_range)):
                # Defining E and R value
                E = E_range[i]
                R = R_range[j]
                timeconst = R/E

                # Defining time period for each equation
                t1 = py.arange(0, tr + dt, dt)
                t2 = py.arange(t1[-1], ti + dt, dt)
                t3 = py.arange(t2[-1], tm + dt , dt)
                t4 = py.arange(t3[-1], tep + dt, dt)
                t5 = py.arange(t4[-1], T , dt)

                Qin = 2*VT/(t2[-1]-t1[0])

                # Defining function handle for each equations for flow rate
                A = Qin/(t1[-1] - t1[0])         # Constant for Q1 and V1
                B = -Qin/(t2[-1]-t2[0])          # Constant for Q2 and V2
            
                
                # Defining function handle for each equations for volume after integration
                C1 = -(A/2)*(t1[0]-t1[-1])**2-Qin*t1[0]  # Constant for V1
                V1 = lambda t: (A/2)*(t-t1[-1])**2 + Qin*t + C1
                C2 = V1(t1[-1]) - (B/2)*(t2[0]-t2[-1])**2            # Constant for V2
                V2 = lambda t: B/2*(t-t2[-1])**2 + C2

                Q1 = lambda t: A*(t-t1[-1]) + Qin  
                Q2 = lambda t: B*(t-t2[-1])

                # Defining Qout
                # For P4 equals to zero
                t_func = lambda t: R*(1-py.exp(-(t3[-1]-t2[-1])/timeconst))*py.exp(-(t-t3[-1])/timeconst)-E*timeconst*py.exp(-(t-t3[-1])/timeconst)*(1-py.exp(-(t3[-1]-t2[-1])/timeconst))+E*timeconst*(1-py.exp(-(t3[-1]-t2[-1])/timeconst))+E*(t3[-1]+timeconst*py.exp(-(t3[-1]-t2[-1])/timeconst))-E*(t2[-1]+timeconst)
                Qout = -E*V2(t2[-1])/t_func(t4[0])
                
                A3 = Qout                                          # Constant for Q3
                Q3 = lambda t: A3*(1-py.exp(-(t-t3[0])/timeconst))
                A4 = Q3(t3[-1])                                    # Constant for Q4
                Q4 = lambda t: A4*py.exp(-(t-t4[0])/timeconst)
                A5 = Q4(t4[-1])/(t4[-1]-t5[-1])**2
                Q5 = lambda t: A5*(t-t5[-1])**2
                C3 = V2(t2[-1]) -A3*(t2[-1] + timeconst)        # Constant for V3
                V3 = lambda t: A3*(t + timeconst*py.exp(-(t-t2[-1])/timeconst)) + C3
                C4 = V3(t3[-1]) + timeconst*A4
                V4 = lambda t: -A4*timeconst*py.exp(-(t-t3[-1])/timeconst) + C4
                C5 = V4(t4[-1]) -(A5/3)*(t4[-1]-t5[-1])**3
                V5 = lambda t: (A5/3)*(t-t5[-1])**3 + C5

                # By implementing linear single compartment equation to calculate pressure
                P1 = E*V1(t1) + R*Q1(t1) + PEEP
                P2 = E*V2(t2) + R*Q2(t2) + PEEP
                P3 = E*V3(t3) + R*Q3(t3) + PEEP
                P4 = E*V4(t4) + R*Q4(t4) + PEEP
                P5 = E*V5(t5) + R*Q5(t5) + PEEP

                # Storing value for t,P,Q, and V
                if i == 0:
                    t = py.concatenate([t1,t2,t3,t4,t5])
                    Q = py.concatenate([Q1(t1),Q2(t2),Q3(t3),Q4(t4),Q5(t5)])
                    V = py.concatenate([V1(t1),V2(t2),V3(t3),V4(t4),V5(t5)])
                    P = py.concatenate([P1,P2,P3,P4,P5])
                else:
                    t_temp = py.concatenate([t1,t2,t3,t4,t5])
                    t = py.concatenate([t,t_temp])
                    Q_temp = py.concatenate([Q1(t1),Q2(t2),Q3(t3),Q4(t4),Q5(t5)])
                    Q = py.concatenate([Q,Q_temp])
                    V_temp = py.concatenate([V1(t1),V2(t2),V3(t3),V4(t4),V5(t5)])
                    V = py.concatenate([V,V_temp])
                    P_temp = py.concatenate([P1,P2,P3,P4,P5])
                    P = py.concatenate([P,P_temp])

                # Obtaining Peak Inspiration Pressure
                PIP = max(P)
                
                # Storing value after each simulation
                T_temp = PIP
                # Saving value to a predifined variables
                T_data.append(PIP)

                # This section save only successful pair
                if PIP <= 30:
                    P_suc.append(round(T_temp,2))
                    E_suc.append(E)
                    R_suc.append(R)


        # Reshaping tidal volume per unit weight in to z-axis
        Z = py.reshape(T_data,(len(R_range),len(E_range)))

        figPairVC = make_subplots(rows=2, cols=1, 
                                vertical_spacing=0.09,
                                row_heights=[0.8,0.2],
                                specs=[[{"type": "contour"}],
                                        [{"type": "table"}]
                                        ])
        # Plotting the contour plots
        figPairVC.add_trace(go.Contour(z=Z, x=E_range, y=R_range, name="VC (Ramp)",
                                            hovertemplate = "E: %{x}"+
                                                            "<br>R: %{y}"+
                                                            "<br>PIP: %{z}",
                                            colorbar_tickfont_size=16,
                                            colorbar_title_font_size=18,
                                            colorbar=dict(
                                                title='<b>PIP (cmH2O)<b>',
                                                titleside='right',
                                                y = 0.63,
                                                len = 0.6
                                                ),  
                                            colorscale=[
                                                # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
                                                [0, "rgb(102, 255, 102)"],
                                                [0.1, "rgb(102, 255, 102)"],

                                                # Let values between 10-20% of the min and max of z
                                                # have color rgb(20, 20, 20)
                                                [0.1, "rgb(102, 255, 102)"],
                                                [0.2, "rgb(102, 255, 102)"],

                                                # Values between 20-30% of the min and max of z
                                                # have color rgb(40, 40, 40)
                                                [0.2, "rgb(102, 255, 102)"],
                                                [0.3, "rgb(102, 255, 102)"],

                                                [0.3, "rgb(102, 255, 102)"],
                                                [0.35, "rgb(178, 255, 102)"],

                                                [0.35, "rgb(178, 255, 102)"],
                                                [0.4, "rgb(178, 255, 102)"],

                                                [0.4, "rgb(178, 255, 102)"],
                                                [0.5, "rgb(178, 255, 102)"],

                                                [0.5, "rgb(178, 255, 102)"],
                                                [0.6, "rgb(178, 255, 102)"],

                                                [0.6, "rgb(178, 255, 102)"],
                                                [0.7, "rgb(255, 255, 102)"],

                                                [0.7, "rgb(255, 255, 102)"],
                                                [0.8, "rgb(255, 178, 102)"],

                                                [0.8, "rgb(255, 178, 102)"],
                                                [0.9, "rgb(255, 102, 102)"],

                                                [0.9, "rgb(255, 102, 102)"],
                                                [1.0, "rgb(255, 102, 102)"]
                                                

                                            ],
                                            contours_coloring='heatmap',
                                            contours=dict(start=0, end=35, size=5, 
                                            showlabels=True, 
                                            labelfont=dict(size=15, color='black')
                                            )
                                            ),
                                            row=1, col=1
                                            )

       
        
        figPairVC.update_xaxes(title_text="<b>Elastance (cmH2O/L)<b>", title_font=dict(size=18))
        figPairVC.update_yaxes(title_text="<b>Resistance (cmH2O.s/L)<b>", title_font=dict(size=18))
        figPairVC.update_layout(title_text="Recommended Patient for Pairing",
                            title_font=dict(size=25),
                            height=1200,
                            hoverlabel=dict(
                                font_size=18
                            )
                                )               
                            
        
        # Creating table for successful pair
        figPairVC.add_trace(go.Table(header=dict(values=['Elastance (cmH2O/L)', 'Resistance (cmH2O.s/L)', 'PIP (cmH2O)'],
                                                font=dict(size=18)
                                                ),
                                    cells=dict(values=[E_suc, R_suc, P_suc],
                                                font=dict(size=16),
                                                height=30
                                                ),
                                                
                                                ),
                                                row=2, col=1
                                                )
        

       
        divPairVC = figPairVC.to_html(full_html=False)
        return divPairVC
       


                
# This function is used to simulate the contour plots in square waveform
class VCSquarePairing:
    def __init__(self, RR, PEEP, VT, I_E):
        # Defining variables for patient
        self.RR = RR            # Respiratory rate (frequency, times/min)
        self.I_E = I_E          # IE ratio
        self.PEEP = PEEP        # PEEP pressure
        self.VT = VT            # Peak Pressure
    
    def simulate(self):
        # Range of Elastance and Resistance to be Simulated (Any Values larger than 0)
        
        #-------Changing step size for R and E--------------
        E_range = py.arange(1,52,5)
        R_range = py.arange(1,52,5)
        #---------------------------------------------------

        RR = self.RR
        I_E = self.I_E
        PEEP = self.PEEP
        VT = (self.VT)/1000

        #-----------------KEPT CONSTANT---------------------------------------
        Rt = 1/10      # Fraction for rise time during inspiration
        Rd = 1/10      # Fraction for drop time during inspiration
        Rp = 1-Rt-Rd   # Fraction for pausing time during inspiration
        Dt = 0.999     # Fraction for exponential curve during expiration
        Mt = 1/3       # Time required to reach minimum flow rate
        #----------------------------------------------------------------------

        # Defining parameters
        T = 1/(RR/60)  # Respiratory Rate (Period)
        ti = T*(1-1/(1+I_E))
        Qin = VT/(ti*(0.5*Rp + 0.5))     # Peak flow (To be calculated based on the input I:E and Tidal Volume)
        te = T-ti                 # Expiration period
        tr = Rt*ti                # Rise time for flow rate during inspiration
        tp = tr + ti*Rp           # End of pausing time
        tee = Dt*te               # Time required to complete exponential curve
        tm = ti + tee*Mt          # Time for dropping exponential curve
        tep = tm + (1-Mt)*tee     # Time for rising exponential curve
        dt = 0.001                # Step size for time

        # Creating empty variable for data storage
        T_data = []
        P_suc = []
        E_suc = []
        R_suc = []

        for j in range(len(R_range)):
            for i in range(len(E_range)):
                # Defining E and R value
                E = E_range[i]
                R = R_range[j]
                timeconst = R/E

                # Defining time period for each equation
                t1 = py.arange(0, tr + dt, dt)
                t2 = py.arange(t1[-1], tp + dt, dt)
                t3 = py.arange(t2[-1], ti + dt , dt)
                t4 = py.arange(t3[-1], tm + dt, dt)
                t5 = py.arange(t4[-1], tep + dt, dt)
                t6 = py.arange(t5[-1], T, dt)

                # Defining function handle for each equations for flow rate
                A1 = Qin/(t1[-1] - t1[0])   # Constant for Q1
                Q1 = lambda t: A1*(t-t1[0])  
                Q2 = lambda t: Qin
                A3 = -Qin/(t3[-1] - t3[0])
                Q3 = lambda t: A3*(t-t3[0]) + Qin
                
                # Defining function handle for each equations for volume after integration
                C1 = 0  # Constant for V1
                V1 = lambda t: A1*(t-t1[0])**2/2 
                C2 = V1(t1[-1]) - Qin * t1[-1]              # Constant for V2
                V2 = lambda t: Qin*t + C2
                C3 = V2(t2[-1]) + A3*(t2[-1]**2/2) - Qin*t2[-1] # Constant for V3
                V3 = lambda t: A3*((t**2)/2-t*t3[0]) + Qin*t + C3

                # Defining Qout
                # For P4 equals to zero
                t_func = lambda t: R*(1-py.exp(-(t4[-1]-t3[-1])/timeconst))*py.exp(-(t-t4[-1])/timeconst)-E*timeconst*py.exp(-(t-t4[-1])/timeconst)*(1-py.exp(-(t4[-1]-t3[-1])/timeconst))+E*timeconst*(1-py.exp(-(t4[-1]-t3[-1])/timeconst))+E*(t4[-1]+timeconst*py.exp(-(t4[-1]-t3[-1])/timeconst))-E*(t3[-1]+timeconst)
                Qout = -E*V3(t3[-1])/t_func(t5[0])
                
                A4 = Qout                                          # Constant for Q3
                Q4 = lambda t: A4*(1-py.exp(-(t-t4[0])/timeconst))
                A5 = Q4(t4[-1])                                    # Constant for Q4
                Q5 = lambda t: A5*py.exp(-(t-t5[0])/timeconst)
                A6 = Q5(t5[-1])/(t5[-1]-t6[-1])**2
                Q6 = lambda t: A6*(t-t6[-1])**2
                C4 = V3(t3[-1]) -A4*(t3[-1] + timeconst)        # Constant for V3
                V4 = lambda t: A4*(t + timeconst*py.exp(-(t-t3[-1])/timeconst)) + C4
                C5 = V4(t4[-1]) + timeconst*A5
                V5 = lambda t: -A5*timeconst*py.exp(-(t-t4[-1])/timeconst) + C5
                C6 = V5(t5[-1]) -(A6/3)*(t5[-1]-t6[-1])**3
                V6 = lambda t: (A6/3)*(t-t6[-1])**3 + C6
            

                # By implementing linear single compartment equation to calculate pressure
                P1 = E*V1(t1) + R*Q1(t1) + PEEP
                P2 = E*V2(t2) + R*Q2(t2) + PEEP
                P3 = E*V3(t3) + R*Q3(t3) + PEEP
                P4 = E*V4(t4) + R*Q4(t4) + PEEP
                P5 = E*V5(t5) + R*Q5(t5) + PEEP
                P6 = E*V6(t6) + R*Q6(t6) + PEEP

                # Storing value for t,P,Q, and V
                if i == 0:
                    t = py.concatenate([t1,t2,t3,t4,t5,t6])
                    Q = py.concatenate([Q1(t1),Q2(t2)*py.ones(len(t2)),Q3(t3),Q4(t4),Q5(t5),Q6(t6)])
                    V = py.concatenate([V1(t1),V2(t2)*py.ones(len(t2)),V3(t3),V4(t4),V5(t5),V6(t6)])
                    P = py.concatenate([P1,P2,P3,P4,P5,P6])
                else:
                    t_temp = py.concatenate([t1,t2,t3,t4,t5,t6])
                    t = py.concatenate([t,t_temp])
                    Q_temp = py.concatenate([Q1(t1),Q2(t2)*py.ones(len(t2)),Q3(t3),Q4(t4),Q5(t5),Q6(t6)])
                    Q = py.concatenate([Q,Q_temp])
                    V_temp = py.concatenate([V1(t1),V2(t2)*py.ones(len(t2)),V3(t3),V4(t4),V5(t5),V6(t6)])
                    V = py.concatenate([V,V_temp])
                    P_temp = py.concatenate([P1,P2,P3,P4,P5,P6])
                    P = py.concatenate([P,P_temp])

                # Obtaining Peak Inspiration Pressure
                PIP = max(P)
                
                # Storing value after each simulation
                T_temp = PIP
                # Saving value to a predifined variables
                T_data.append(PIP)

                # This section save only successful pair
                if PIP <= 30:
                    P_suc.append(round(T_temp,2))
                    E_suc.append(E)
                    R_suc.append(R)

        # Reshaping tidal volume per unit weight in to z-axis
        Z = py.reshape(T_data,(len(R_range),len(E_range)))

        figPairVC = make_subplots(rows=2, cols=1, 
                                vertical_spacing=0.09,
                                row_heights=[0.8,0.2],
                                specs=[[{"type": "contour"}],
                                        [{"type": "table"}]
                                        ])
        # Plotting the cntour plots
        figPairVC.add_trace(go.Contour(z=Z, x=E_range, y=R_range, name="VC (Square)",
                                            hovertemplate = "E: %{x}"+
                                                            "<br>R: %{y}"+
                                                            "<br>PIP: %{z}",
                                            colorbar_tickfont_size=16,
                                            colorbar_title_font_size=18,
                                            colorbar=dict(
                                                title='<b>PIP (cmH2O)<b>',
                                                titleside='right',
                                                y = 0.63,
                                                len = 0.6
                                                ),  
                                            colorscale=[
                                                # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
                                                [0, "rgb(102, 255, 102)"],
                                                [0.1, "rgb(102, 255, 102)"],

                                                # Let values between 10-20% of the min and max of z
                                                # have color rgb(20, 20, 20)
                                                [0.1, "rgb(102, 255, 102)"],
                                                [0.2, "rgb(102, 255, 102)"],

                                                # Values between 20-30% of the min and max of z
                                                # have color rgb(40, 40, 40)
                                                [0.2, "rgb(102, 255, 102)"],
                                                [0.3, "rgb(102, 255, 102)"],

                                                [0.3, "rgb(102, 255, 102)"],
                                                [0.35, "rgb(178, 255, 102)"],

                                                [0.35, "rgb(178, 255, 102)"],
                                                [0.4, "rgb(178, 255, 102)"],

                                                [0.4, "rgb(178, 255, 102)"],
                                                [0.5, "rgb(178, 255, 102)"],

                                                [0.5, "rgb(178, 255, 102)"],
                                                [0.6, "rgb(178, 255, 102)"],

                                                [0.6, "rgb(178, 255, 102)"],
                                                [0.7, "rgb(255, 255, 102)"],

                                                [0.7, "rgb(255, 255, 102)"],
                                                [0.8, "rgb(255, 178, 102)"],

                                                [0.8, "rgb(255, 178, 102)"],
                                                [0.9, "rgb(255, 102, 102)"],

                                                [0.9, "rgb(255, 102, 102)"],
                                                [1.0, "rgb(255, 102, 102)"]
                                                

                                            ],
                                            contours_coloring='heatmap',
                                            contours=dict(start=0, end=35, size=5, 
                                            showlabels=True, 
                                            labelfont=dict(size=18, color='black')
                                            )
                                            ),
                                            row=1, col=1
                                            )
        

       
        figPairVC.update_xaxes(title_text="<b>Elastance (cmH2O/L)<b>", title_font=dict(size=18))
        figPairVC.update_yaxes(title_text="<b>Resistance (cmH2O.s/L)<b>", title_font=dict(size=18))
        figPairVC.update_layout(title_text="Recommended Patient for Pairing",
                            title_font=dict(size=25),
                            height=1200,
                            hoverlabel=dict(
                                font_size=18
                            )
                                )               
                            
        
        # Creating table for successful pair

        figPairVC.add_trace(go.Table(header=dict(values=['Elastance (cmH2O/L)', 'Resistance (cmH2O.s/L)', 'PIP (cmH2O)'],
                                                font=dict(size=18)
                                                ),
                                    cells=dict(values=[E_suc, R_suc, P_suc],
                                                font=dict(size=16),
                                                height=30
                                                ),
                                                
                                                ),
                                                row=2, col=1
                                                )
        

       
        divPairVC = figPairVC.to_html(full_html=False)
        return divPairVC
        





