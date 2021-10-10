import numpy as py
import plotly.graph_objects as go
from scipy.integrate import odeint, cumtrapz
from plotly.subplots import make_subplots
from statistics import mean

# This function is used to simulate the estimated ventilation graphs for the current patient


class PCModel:

    def __init__(self, Weight, E, R, RR, PEEP, PIP, I_E):
        # Defining variables for patient
        self.Weight = Weight
        self.E = E
        self.R = R
        self.RR = RR         # Respiratory rate (frequency, times/min)
        self.I_E = I_E       # IE ratio
        self.PEEP = PEEP     # PEEP pressure
        self.PIP = PIP       # Peak Pressure

    def simulate(self):
        # Defining settings for ventilator
        E = self.E
        R = self.R
        RR = self.RR
        I_E = self.I_E
        PEEP = self.PEEP
        PIP = self.PIP
        Weight = self.Weight

        # --------------KEPT CONSTANT---------------------
        Rti = 0.2       # Rise time ratio for inspiration
        Rto = 0.99      # Pausing ratio during expiration
        # ------------------------------------------------

        T = 1/(RR/60)   # Respiratory Rate (Period, second)

        # Defining time for each respiratory stage
        ti = (1-1/(1+I_E))*T                # Total inspiration period
        tr = ti*Rti                         # Rising time
        te = ti + (1/(1+I_E))*T*(1-Rto)     # Expiration time

        Ti = 5                              # Number of desired periods

        # Defining stepsize values
        dt = 0.001

        # -----------------Constructing equation for P(t), V(t) and Q(t)-----------------------------------------
        # Defining time range
        t1 = py.arange(0, tr + dt, dt)
        t2 = py.arange(t1[-1], ti + dt, dt)
        t3 = py.arange(t2[-1], te + dt, dt)
        t4 = py.arange(t3[-1], T, dt)

        # Defining contants for each equations
        A = (PIP-PEEP)/(t1[-1] - t1[0])
        B = (PEEP-PIP)/(t3[-1] - t3[0])

        # Defining equation of pressure for each time
        def P1(t): return A*(t-t1[-1]) + PIP
        def P2(t): return PIP
        def P3(t): return B*(t-t3[-1]) + PEEP
        def P4(t): return PEEP

        # Defining the function for first order ODE:
        def dv_dt1(v, t): return (P1(t) - PEEP - E*v)/R
        def dv_dt2(v, t): return (P2(t) - PEEP - E*v)/R
        def dv_dt3(v, t): return (P3(t) - PEEP - E*v)/R
        def dv_dt4(v, t): return (P4(t) - PEEP - E*v)/R

        # Solving the first order ODE for Volume and combining into one array
        v1 = py.concatenate(odeint(dv_dt1, 0, t1))
        v2 = py.concatenate(odeint(dv_dt2, v1[-1], t2))
        v3 = py.concatenate(odeint(dv_dt3, v2[-1], t3))
        v4 = py.concatenate(odeint(dv_dt4, v3[-1], t4))

        # Solving the flow from the computed volume (Applying axis=0 to indicate gradient in only one direction)
        Q1 = py.gradient(v1, dt, axis=0, edge_order=2)
        Q2 = py.gradient(v2, dt, axis=0, edge_order=2)
        Q3 = py.gradient(v3, dt, axis=0, edge_order=2)
        Q4 = py.gradient(v4, dt, axis=0, edge_order=2)

        # Storing value for t,P,Q, and V
        t = py.concatenate([t1, t2, t3, t4])
        Q = py.concatenate([Q1, Q2, Q3, Q4])
        V = py.concatenate([v1, v2, v3, v4])
        P = py.concatenate([P1(t1), P2(t2)*py.ones(len(t2)),
                           P3(t3), P4(t4)*py.ones(len(t4))])

        # This part duplicate the previous results to plot more than one periods of breathing cycles
        for i in range(1, Ti):
            # Defining time range
            t1 = py.arange(i*T, tr + T*i + dt, dt)
            t2 = py.arange(t1[-1], ti + T*i + dt, dt)
            t3 = py.arange(t2[-1], te + T*i + dt, dt)
            t4 = py.arange(t3[-1], T + T*i, dt)

            t_temp = py.concatenate([t1, t2, t3, t4])
            t = py.concatenate([t, t_temp])
            Q_temp = Q
            Q = py.concatenate([Q, Q_temp])
            V_temp = V
            V = py.concatenate([V, V_temp])
            P_temp = P
            P = py.concatenate([P, P_temp])

        # Tidal volume for the current patient
        VT = round(max(V)*1000/Weight, 2)
        Qin = round(max(Q)*60, 2)  # Peak flow rate for the current patient
        # ----------------------------End of graph construction-----------------------------------------------------------------------------------

        # Creating subplots using plotly
        figP1 = make_subplots(rows=4, cols=1,
                              vertical_spacing=0.1,
                              specs=[[{"type": "scatter"}],
                                     [{"type": "scatter"}],
                                     [{"type": "scatter"}],
                                     [{"type": "table"}],
                                     ])

        # Plotting the graph using plotly for pressure, flow, and volume
        figP1.add_trace(go.Scatter(x=t, y=P, name="Pressure"), row=1, col=1)
        figP1.add_trace(go.Scatter(x=t, y=Q, name="Flow"), row=2, col=1)
        figP1.add_trace(go.Scatter(x=t, y=V, name="Volume"), row=3, col=1)

        # Summarise on the MV setting
        figP1.add_trace(go.Table(header=dict(values=['PIP (cmH2O)', 'PEEP (cmH2O)', 'RR (1/min)', 'I:E', 'VT (mL/kg)', 'Qin (L/min)'],
                                             font=dict(size=16)
                                             ),
                                 cells=dict(values=[[PIP], [PEEP], [RR], [I_E], [VT], [Qin]],
                                            font=dict(size=16),
                                            height=30
                                            ),
                                 ),
                        row=4, col=1,
                        )

        # Update xaxis properties
        figP1.update_xaxes(title_text="<b>Time(s)<b>", title_font=dict(
            size=18), row=3, col=1)  # Similar for all graph

        # Update yaxis properties
        figP1.update_yaxes(title_text="<b>Pressure (cmH2O)<b>",
                           title_font=dict(size=15), row=1, col=1)
        figP1.update_yaxes(title_text="<b>Flow (L/s)<b>",
                           title_font=dict(size=15),  row=2, col=1)
        figP1.update_yaxes(title_text="<b>Volume (L)<b>",
                           title_font=dict(size=15),  row=3, col=1)

        # Updating figure layout
        figP1.update_layout(showlegend=False, title_text="Estimated Ventilator Graph for Current Patient",
                            title_font=dict(size=25),
                            height=800,
                            hoverlabel=dict(
                                font_size=18
                            )
                            )
        divP1 = figP1.to_html(full_html=False)
        return divP1

# This function is used to simulate the contour plots for each pairing patient with different lung mechanics and weights


class PCPairing:
    def __init__(self, W, E2, R2,  RR, PEEP, PIP, I_E):
        self.W = W
        self.RR = RR
        self.PEEP = PEEP
        self.PIP = PIP
        self.I_E = I_E
        self.E2 = E2
        self.R2 = R2

    def simulate(self):

        # Range of Elastance and Resistance to be Simulated (Any Values larger than 0)
        # -----------------Step size can be changed here------------------------------------
        E_range = py.arange(1, 52, 5)
        R_range = py.arange(1, 52, 5)
        # ----------------------------------------------------------------------------------

        Weight = self.W
        E2 = self.E2
        R2 = self.R2
        Weight, E2, R2 = zip(*sorted(zip(Weight, E2, R2)))

        # Defining settings for ventilator
        # -----------------------KEEP CONSTANT------------------------------------------
        Rti = 0.2       # Rise time ratio for inspiration
        Rto = 0.99      # Pausing ratio during expiration
        # -------------------------------------------------------------------------------

        I_E = self.I_E      # IE ratio
        RR = self.RR        # Respiratory rate (frequency, times/min)
        T = 1/(RR/60)       # Respiratory Rate (Period, second)
        PEEP = self.PEEP    # PEEP pressure
        PIP = self.PIP      # Peak Pressure

        # Defining stepsize values
        dt = 0.001

        # Creating empty array to store the values
        T_data = []
        E_corr = []
        R_corr = []
        V_suc = []
        E_suc = []
        R_suc = []

        # Defining time for each respiratory stage
        ti = (1-1/(1+I_E))*T                # Total inspiration period
        tr = round(ti*Rti, 2)                # Rising time
        te = ti + (1/(1+I_E))*T*(1-Rto)     # Expiration time

        # This section starts simulation with different combinations of lung mechanics
        for j in range(len(R_range)):
            for i in range(len(E_range)):
                # Defining E and R value
                E = E_range[i]
                R = R_range[j]

                # Defining time range
                t1 = py.arange(0, tr + dt, dt)
                t2 = py.arange(t1[-1], ti + dt, dt)
                t3 = py.arange(t2[-1], te + dt, dt)
                t4 = py.arange(t3[-1], T, dt)

                # Defining contants for each equations
                A = (PIP-PEEP)/(t1[-1] - t1[0])
                B = (PEEP-PIP)/(t3[-1] - t3[0])

                # Defining equation of pressure for each time
                def P1(t): return A*(t-t1[-1]) + PIP
                def P2(t): return PIP
                def P3(t): return B*(t-t3[-1]) + PEEP
                def P4(t): return PEEP

                # Defining the function for first order ODE:
                def dv_dt1(v, t): return (P1(t) - PEEP - E*v)/R
                def dv_dt2(v, t): return (P2(t) - PEEP - E*v)/R
                def dv_dt3(v, t): return (P3(t) - PEEP - E*v)/R
                def dv_dt4(v, t): return (P4(t) - PEEP - E*v)/R

                # Solving the first order ODE for Volume and combining into one array
                v1 = py.concatenate(odeint(dv_dt1, 0, t1))
                v2 = py.concatenate(odeint(dv_dt2, v1[-1], t2))
                v3 = py.concatenate(odeint(dv_dt3, v2[-1], t3))
                v4 = py.concatenate(odeint(dv_dt4, v3[-1], t4))

                # Solving the flow from the computed volume (Applying axis=0 to indicate gradient in only one direction)
                Q1 = py.gradient(v1, dt, axis=0, edge_order=2)
                Q2 = py.gradient(v2, dt, axis=0, edge_order=2)
                Q3 = py.gradient(v3, dt, axis=0, edge_order=2)
                Q4 = py.gradient(v4, dt, axis=0, edge_order=2)

                # Storing value for t,P,Q, and V
                t = py.concatenate([t1, t2, t3, t4])
                Q = py.concatenate([Q1, Q2, Q3, Q4])
                V = py.concatenate([v1, v2, v3, v4])
                P = py.concatenate(
                    [P1(t1), P2(t2)*py.ones(len(t2)), P3(t3), P4(t4)*py.ones(len(t4))])

                # Obtaining Tidal Volume
                VT = max(V)
                VT_mass = []

                # Storing value after each simulation
                T_temp = VT

                # Saving value to a predifined variables
                T_data.append(T_temp)
                E_corr.append(E)
                R_corr.append(R)

        VT_mass = []

        # Converting list to array
        T_data = py.array(T_data)
        Weight = py.array(Weight)

        # This section starts simulate tidal volume data for patients with different weights
        for k in range(len(Weight)):
            V_temp = []
            E_temp = []
            R_temp = []

            # Obtaining Tidal Volume per kg
            VT_temp = T_data*1000/Weight[k]

            for L in range(len(VT_temp)):
                Test_VT = VT_temp[L]
                # This section save only successful pair
                if Test_VT >= 6 and Test_VT <= 8:
                    V_temp.append(round(Test_VT, 2))
                    E_temp.append(E_corr[L])
                    R_temp.append(R_corr[L])

            VT_mass.append(VT_temp)

            # Rearranging
            if V_temp:
                E_temp, R_temp, V_temp = zip(
                    *sorted(zip(E_temp, R_temp, V_temp)))
            else:  # No combination of lung mechanics is able to achieved the predefined criteria
                E_temp = "No data"
                R_temp = "No data"
                V_temp = "No data"

            V_suc.append(V_temp)
            E_suc.append(E_temp)
            R_suc.append(R_temp)

        # --------------------------Plotting graphs based on the simulated data from prvious part-------------------------------

        # Defined variables for iteration increments in the following sections
        iter = -1
        iter2 = -1
        iter3 = -1

        figPair = make_subplots(rows=2, cols=1,
                                vertical_spacing=0.09,
                                row_heights=[0.8, 0.2],
                                specs=[[{"type": "contour"}],
                                       [{"type": "table"}]
                                       ])

        # Plotting the contour plots with different weight
        for step in Weight:
            iter = iter + 1
            # Reshaping tidal volume per unit weight in to z-axis
            Z = py.reshape(VT_mass[iter], (len(R_range), len(E_range)))

            # Plotting the cntour plots
            figPair.add_trace(go.Contour(z=Z, x=E_range, y=R_range,
                                         visible=False,
                                         name="PC Pairing",
                                         hovertemplate="E: %{x}" +
                                         "<br>R: %{y}" +
                                         "<br>TV: %{z}",
                                         colorbar=dict(
                                             title='<b>Tidal Volume (mL/kg)<b>',
                                             titleside='right',
                                             y=0.63,
                                             len=0.6
                                         ),
                                         colorbar_tickfont_size=16,
                                         colorbar_title_font_size=18,
                                         colorscale=[
                                             # Let first 10% (0.1) of the values have color rgb(0, 0, 0)
                                                    [0, "rgb(255, 102, 102)"],
                                                    [0.1,
                                                        "rgb(255, 102, 102)"],

                                             # Let values between 10-20% of the min and max of z
                                             # have color rgb(20, 20, 20)
                                                    [0.1,
                                                        "rgb(255, 102, 102)"],
                                                    [0.2,
                                                        "rgb(255, 178, 102)"],

                                             # Values between 20-30% of the min and max of z
                                             # have color rgb(40, 40, 40)
                                                    [0.2,
                                                        "rgb(255, 178, 102)"],
                                                    [0.3,
                                                        "rgb(255, 178, 102)"],

                                                    [0.3,
                                                        "rgb(255, 178, 102)"],
                                                    [0.35,
                                                        "rgb(255, 178, 102)"],

                                                    [0.35,
                                                        "rgb(255, 178, 102)"],
                                                    [0.4,
                                                        "rgb(255, 255, 102)"],

                                                    [0.4,
                                                        "rgb(255, 255, 102)"],
                                                    [0.5,
                                                        "rgb(255, 255, 102)"],

                                                    [0.5,
                                                        "rgb(255, 255, 102)"],
                                                    [0.6,
                                                        "rgb(178, 255, 102)"],

                                                    [0.6,
                                                        "rgb(178, 255, 102)"],
                                                    [0.7,
                                                        "rgb(178, 255, 102)"],

                                                    [0.7,
                                                        "rgb(178, 255, 102)"],
                                                    [0.8,
                                                        "rgb(255, 255, 102)"],

                                                    [0.8,
                                                        "rgb(255, 255, 102)"],
                                                    [0.9,
                                                        "rgb(255, 178, 102)"],

                                                    [0.9,
                                                        "rgb(255, 178, 102)"],
                                                    [1.0, "rgb(255, 102, 102)"]




                                         ],
                                         contours_coloring='heatmap',
                                         contours=dict(start=0, end=10, size=2,
                                                       showlabels=True,
                                                       labelfont=dict(
                                                           size=18, color='black')
                                                       ),
                                         ),
                              row=1, col=1
                              )

        # Plotting the rhombus point based on the lung mechanics of the patients
        for step in Weight:
            iter2 = iter2+1
            figPair.add_trace(go.Scatter(name='Patient', x=[E2[iter2]], y=[R2[iter2]],
                                         visible=False,
                                         marker=dict(
                                             color="black", size=15, symbol="diamond"),
                                         mode="markers",
                                         ),
                              row=1, col=1
                              )

        # Creating table for contour plots with different weights
        for step in Weight:
            iter3 = iter3+1
            # Creating table for successful pair

            figPair.add_trace(go.Table(header=dict(values=['Elastance (cmH2O/L)', 'Resistance (cmH2O.s/L)', 'Tidal Volume (mL/kg)'],
                                                   font=dict(size=18)
                                                   ),
                                       visible=False,
                                       cells=dict(values=[E_suc[iter3], R_suc[iter3], V_suc[iter3]],
                                                  font=dict(size=16),
                                                  height=30
                                                  )
                                       ),
                              row=2, col=1
                              )

        figPair.data[0].visible = True
        figPair.data[0+len(Weight)].visible = True
        figPair.data[0+len(Weight)*2].visible = True

        # Create and add slider
        steps = []
        for i in range(len(Weight)):
            W = str(Weight[i])
            step = dict(
                method="update",
                args=[{"visible": [False] * len(Weight)},
                      {"title": "Pairing Patient Weight: " + str(Weight[i]) + " kg"}],  # layout attribute
                label=W + " kg"
            )
            # Toggle i'th trace to "visible"
            step["args"][0]["visible"][i] = True
            steps.append(step)

        sliders = [dict(font={'size': 20},
                        active=10,
                        currentvalue={"prefix": "Weight: "},
                        pad={"t": 50},
                        steps=steps
                        )]

        figPair.update_layout(
            sliders=sliders
        )

        figPair.update_xaxes(
            title_text="<b>Elastance (cmH2O/L)<b>", title_font=dict(size=18))
        figPair.update_yaxes(
            title_text="<b>Resistance (cmH2O.s/L)<b>", title_font=dict(size=18))
        figPair.update_layout(title="Recommended Pairing Patient", title_font=dict(size=25),
                              height=1200,
                              hoverlabel=dict(
            font_size=18
        )
        )
        figPair.update_layout(yaxis=dict(
            tickfont=dict(size=20)),
            xaxis=dict(
            tickfont=dict(size=20))
        )

        figPair.update_layout(legend=dict(
            orientation="h",

            font=dict(

                size=20,

            ),
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ))

        figPair.update_layout(showlegend=True)

        divPair = figPair.to_html(full_html=False)
        return divPair


# This function uses double compartment model to estimate the ideal MV setting for co-ventilation
class PCSetting:

    def __init__(self, W1, W2, E1, R1, E2, R2, Rc, RR, PEEP, PIP, I_E):
        self.W1 = W1
        self.W2 = W2
        self.RR = RR
        self.PEEP = PEEP
        self.PIP = PIP
        self.I_E = I_E
        self.E1 = E1
        self.R1 = R1
        self.E2 = E2
        self.R2 = R2
        self.Rc = Rc

    def simulate(self):
        W1 = self.W1
        W2 = self.W2
        RR = self.RR
        PEEP = self.PEEP
        PIP = self.PIP
        I_E = self.I_E
        E1 = self.E1
        R1 = self.R1
        E2 = self.E2
        R2 = self.R2
        Rc = self.Rc

        # Defining Constant Value for Pressure Graph
        # -----------------KEPT CONSTANT---------------
        Rti = 0.2   # Pausing ratio during inspiration
        Rto = 0.99  # Pausing ratio during expiration
        # ---------------------------------------------

        T = 1/(RR/60)
        R = R1 + R2
        E = E1 + E2
        timeconst1 = R1/E1
        timeconst2 = R2/E2

        # Creating Empty Variables for storage
        VT1 = []
        VT2 = []

        # Defining parameters
        ti = (1-1/(1+I_E))*T
        tr = ti*Rti
        te = ti + (1/(1+I_E))*T*(1-Rto)

        # Defining stepsize values
        dt = 0.001
        Ti = 3

        # -----------------Equation for common pressure Pj(t)-------------
        # Defining time range
        t1 = py.arange(0, tr + dt, dt)
        t2 = py.arange(t1[-1], ti + dt, dt)
        t3 = py.arange(t2[-1], te + dt, dt)
        t4 = py.arange(t3[-1], T, dt)

        t = py.concatenate([t1, t2, t3, t4])
        # Defining constants for each equations
        dP = PIP-PEEP
        A = dP/(t1[-1] - t1[0])
        B = -dP/(t3[-1] - t3[0])

        # Defining equation of pressure for each time
        def Pj1(t): return A*(t-t1[-1]) + PIP
        def Pj2(t): return PIP
        def Pj3(t): return B*(t-t3[-1]) + PEEP
        def Pj4(t): return PEEP

        # Solving for the corresponding Flow and Volume analytically
        # ----------------------------------Patient 1------------------------------------------------------------
        # Defining function handle for Volume
        C11 = ((A/E1)*(-t1[0]+t1[-1]+timeconst1) - dP/E1) * \
            py.exp(t1[0]/timeconst1)
        def V1P1(t): return dP/E1 + (A/E1) * \
            (t-t1[-1]) - A*timeconst1/E1 + C11*py.exp(-t/timeconst1)
        C21 = (V1P1(t1[-1]) - dP*timeconst1/R1) * py.exp(t1[-1]/timeconst1)
        def V2P1(t): return dP*timeconst1/R1 + C21*py.exp(-t/timeconst1)
        C31 = (R1*V2P1(t2[-1]) - B*timeconst1 *
               (t2[-1]-t3[-1]-timeconst1))*py.exp(t2[-1]/timeconst1)
        def V3P1(t): return (B*timeconst1 *
                             (t-t3[-1]-timeconst1) + C31*py.exp(-t/timeconst1))/R1
        C41 = V3P1(t3[-1])*py.exp(t3[-1]/timeconst1)
        def V4P1(t): return C41*py.exp(-t/timeconst1)

        # Defining function handle for Flow
        def Q1P1(t): return A/E1 - C11*py.exp(-t/timeconst1)/timeconst1
        def Q2P1(t): return - C21*py.exp(-t/timeconst1)/timeconst1
        def Q3P1(t): return (B*timeconst1 - C31 *
                             py.exp(-t/timeconst1)/timeconst1)/R1

        def Q4P1(t): return -C41*py.exp(-t/timeconst1)/timeconst1

        VactP1 = py.concatenate([V1P1(t1), V2P1(t2), V3P1(t3), V4P1(t4)])
        QactP1 = py.concatenate([Q1P1(t1), Q2P1(t2), Q3P1(t3), Q4P1(t4)])

        # --------------------------------Patient 2------------------------------------------------------------
        # Defining function handle for volume
        C12 = ((A/E2)*(-t1[0]+t1[-1]+timeconst2) - dP/E2) * \
            py.exp(t1[0]/timeconst2)
        def V1P2(t): return dP/E2 + (A/E2) * \
            (t-t1[-1]) - A*timeconst2/E2 + C12*py.exp(-t/timeconst2)
        C22 = (V1P2(t1[-1]) - dP*timeconst2/R2) * py.exp(t1[-1]/timeconst2)
        def V2P2(t): return dP*timeconst2/R2 + C22*py.exp(-t/timeconst2)
        C32 = (R2*V2P2(t2[-1]) - B*timeconst2 *
               (t2[-1]-t3[-1]-timeconst2))*py.exp(t2[-1]/timeconst2)
        def V3P2(t): return (B*timeconst2 *
                             (t-t3[-1]-timeconst2) + C32*py.exp(-t/timeconst2))/R2
        C42 = V3P2(t3[-1])*py.exp(t3[-1]/timeconst2)
        def V4P2(t): return C42*py.exp(-t/timeconst2)

        # Defining function handle for Flow
        def Q1P2(t): return A/E2 - C12*py.exp(-t/timeconst2)/timeconst2
        def Q2P2(t): return - C22*py.exp(-t/timeconst2)/timeconst2
        def Q3P2(t): return (B*timeconst2 - C32 *
                             py.exp(-t/timeconst2)/timeconst2)/R2

        def Q4P2(t): return -C42*py.exp(-t/timeconst2)/timeconst2

        VactP2 = py.concatenate([V1P2(t1), V2P2(t2), V3P2(t3), V4P2(t4)])
        QactP2 = py.concatenate([Q1P2(t1), Q2P2(t2), Q3P2(t3), Q4P2(t4)])

        # From the desired data of two patients, we will now figure out the required ventilator setting with a predefined value of Rc
        # -------------------Equation of dQdt for solving second order ODE----------------------
        # Patient 1
        def dQ1_dtP1(t): return (1/(timeconst1)**2)*C11*py.exp(-t/timeconst1)
        def dQ2_dtP1(t): return (1/(timeconst1)**2)*C21*py.exp(-t/timeconst1)
        def dQ3_dtP1(t): return (1/(R1*(timeconst1)**2)) * \
            C31*py.exp(-t/timeconst1)

        def dQ4_dtP1(t): return (1/(timeconst1)**2)*C41*py.exp(-t/timeconst1)

        # Patient 2
        def dQ1_dtP2(t): return (1/(timeconst2)**2)*C12*py.exp(-t/timeconst2)
        def dQ2_dtP2(t): return (1/(timeconst2)**2)*C22*py.exp(-t/timeconst2)
        def dQ3_dtP2(t): return (1/(R2*(timeconst2)**2)) * \
            C32*py.exp(-t/timeconst2)

        def dQ4_dtP2(t): return (1/(timeconst2)**2)*C42*py.exp(-t/timeconst2)

        # Pressure (minimum setting on ventilator)
        def P1(t): return E1*V1P1(t) + (R1+Rc)*Q1P1(t) + Rc*Q1P2(t) + PEEP
        def P2(t): return E1*V2P1(t) + (R1+Rc)*Q2P1(t) + Rc*Q2P2(t) + PEEP
        def P3(t): return E1*V3P1(t) + (R1+Rc)*Q3P1(t) + Rc*Q3P2(t) + PEEP
        def P4(t): return E1*V4P1(t) + (R1+Rc)*Q4P1(t) + Rc*Q4P2(t) + PEEP

        # dpdt
        def dP1_dt(t): return E1*Q1P1(t) + (R1+Rc)*dQ1_dtP1(t) + Rc*dQ1_dtP2(t)
        def dP2_dt(t): return E1*Q2P1(t) + (R1+Rc)*dQ2_dtP1(t) + Rc*dQ2_dtP2(t)
        def dP3_dt(t): return E1*Q3P1(t) + (R1+Rc)*dQ3_dtP1(t) + Rc*dQ3_dtP2(t)
        def dP4_dt(t): return E1*Q4P1(t) + (R1+Rc)*dQ4_dtP1(t) + Rc*dQ4_dtP2(t)

        # Defining the function for second order ODE:
        def f1(x, t): return [x[1], (R*dP1_dt(t) + E*(P1(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

        def f2(x, t): return [x[1], (R*dP2_dt(t) + E*(P2(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

        def f3(x, t): return [x[1], (R*dP3_dt(t) + E*(P3(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]
        def f4(x, t): return [x[1], (R*dP4_dt(t) + E*(P4(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

        # Defining the initial conditions for first period
        vo = [0, 0]

        # Solving the second order ODE for Volume and Flow
        v1 = odeint(f1, vo, t1)
        v2 = odeint(f2, [v1[-1, 0], v1[-1, 1]], t2)
        v3 = odeint(f3, [v2[-1, 0], v2[-1, 1]], t3)
        v4 = odeint(f4, [v3[-1, 0], v3[-1, 1]], t4)

        # Obtaining values for flow rate
        Q1 = v1[:, 1]
        Q2 = v2[:, 1]
        Q3 = v3[:, 1]
        Q4 = v4[:, 1]

        # Obtaining values for volume
        V1 = v1[:, 0]
        V2 = v2[:, 0]
        V3 = v3[:, 0]
        V4 = v4[:, 0]

        # Now we generate the ventilator setting based on the desired output
        # Collecting data for the ventilator setting
        # Minimum pressure to obtain the desired result (Obtained from the ramp section)
        Pmax = round(mean(P2(t2)), 0)

        # Constructing the graph that can be generated by ventilator (Square wave instead of ramp wave)
        DP = Pmax - PEEP
        X = DP/(t1[-1] - t1[0])
        Y = -DP/(t3[-1] - t3[0])

        def Pv1(t): return X*(t-t1[-1]) + Pmax
        def Pv2(t): return Pmax
        def Pv3(t): return Y*(t-t3[-1]) + PEEP
        def Pv4(t): return PEEP

        def dPv1_dt(t): return X
        def dPv2_dt(t): return 0
        def dPv3_dt(t): return Y
        def dPv4_dt(t): return 0

        # Defining the function for second order ODE:
        def g1(x, t): return [x[1], (R*dPv1_dt(t) + E*(Pv1(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

        def g2(x, t): return [x[1], (R*dPv2_dt(t) + E*(Pv2(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

        def g3(x, t): return [x[1], (R*dPv3_dt(t) + E*(Pv3(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

        def g4(x, t): return [x[1], (R*dPv4_dt(t) + E*(Pv4(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

        # Solving the second order ODE for Volume and Flow
        vv1 = odeint(g1, vo, t1)
        vv2 = odeint(g2, [vv1[-1, 0], vv1[-1, 1]], t2)
        vv3 = odeint(g3, [vv2[-1, 0], vv2[-1, 1]], t3)
        vv4 = odeint(g4, [vv3[-1, 0], vv3[-1, 1]], t4)

        # Obtaining value for flow rate
        Qv1 = vv1[:, 1]
        Qv2 = vv2[:, 1]
        Qv3 = vv3[:, 1]
        Qv4 = vv4[:, 1]

        # Obtaining value for volume
        Vv1 = vv1[:, 0]
        Vv2 = vv2[:, 0]
        Vv3 = vv3[:, 0]
        Vv4 = vv4[:, 0]

        # Obtaining junction pressure
        Pi1 = Pv1(t1) - Rc*Qv1
        Pi2 = Pv2(t2)*py.ones(len(t2)) - Rc*Qv2
        Pi3 = Pv3(t3) - Rc*Qv3
        Pi4 = Pv4(t4)*py.ones(len(t4)) - Rc*Qv4

        # --------------------------------Creating function for Picard Iteration-------------------------------------
        def picardI(fequation, prange, xrange, yo, precision):
            error = 1
            ytemp = yo + \
                cumtrapz(fequation(prange, py.ones(
                    len(xrange))*yo), xrange, initial=0)
            i = 0

            while error >= precision:
                y = yo + cumtrapz(fequation(prange, ytemp), xrange, initial=0)
                error_temp = abs(y-ytemp)
                error = max(error_temp)
                ytemp = y
                i = i+1
                if i > 1000:
                    precision = error*5

            return y
        # ------------------------------------------------------------------------------------------------------------

        # Using Picard Iteration to solve for volume in each patient
        # ---------------------------------------------Patient 1----------------------------------------------------
        def dv_dt(P, v): return (P - PEEP - E1*v)/R1
        Vo = 0
        V1Pa1 = picardI(dv_dt, Pi1, t1, Vo, dt)
        V2Pa1 = picardI(dv_dt, Pi2, t2, V1Pa1[-1], dt)
        V3Pa1 = picardI(dv_dt, Pi3, t3, V2Pa1[-1], dt)
        V4Pa1 = picardI(dv_dt, Pi4, t4, V3Pa1[-1], dt)

        Q1Pa1 = py.gradient(V1Pa1, dt, axis=0, edge_order=2)
        Q2Pa1 = py.gradient(V2Pa1, dt, axis=0, edge_order=2)
        Q3Pa1 = py.gradient(V3Pa1, dt, axis=0, edge_order=2)
        Q4Pa1 = py.gradient(V4Pa1, dt, axis=0, edge_order=2)

        VsimP1 = py.concatenate([V1Pa1, V2Pa1, V3Pa1, V4Pa1])
        QsimP1 = py.concatenate([Q1Pa1, Q2Pa1, Q3Pa1, Q4Pa1])

        # --------------------------------Patient 2----------------------------------------------------
        def dv_dt(P, v): return (P - PEEP - E2*v)/R2
        V1Pa2 = picardI(dv_dt, Pi1, t1, Vo, dt)
        V2Pa2 = picardI(dv_dt, Pi2, t2, V1Pa2[-1], dt)
        V3Pa2 = picardI(dv_dt, Pi3, t3, V2Pa2[-1], dt)
        V4Pa2 = picardI(dv_dt, Pi4, t4, V3Pa2[-1], dt)

        Q1Pa2 = py.gradient(V1Pa2, dt, axis=0, edge_order=2)
        Q2Pa2 = py.gradient(V2Pa2, dt, axis=0, edge_order=2)
        Q3Pa2 = py.gradient(V3Pa2, dt, axis=0, edge_order=2)
        Q4Pa2 = py.gradient(V4Pa2, dt, axis=0, edge_order=2)

        VsimP2 = py.concatenate([V1Pa2, V2Pa2, V3Pa2, V4Pa2])
        QsimP2 = py.concatenate([Q1Pa2, Q2Pa2, Q3Pa2, Q4Pa2])

        # Obtaining the error of tidal volume
        VTact1 = max(VactP1)
        VTsim1 = max(VsimP1)
        VT1_err = (abs(VTsim1 - VTact1) / VTact1)*100

        VTact2 = max(VactP2)
        VTsim2 = max(VsimP2)
        VT2_err = (abs(VTsim2 - VTact2) / VTact2)*100

        # Storing the value
        VT1 = VT1_err
        VT2 = VT2_err

        # Seeking for a suitable PIP setting with minimum error
        while VT1 and VT2 > 1:
            Pmax = Pmax + 0.5

            # Constructing the graph that can be generated by ventilator
            DP = Pmax - PEEP
            X = DP/(t1[-1] - t1[0])
            Y = -DP/(t3[-1] - t3[0])

            def Pv1(t): return X*(t-t1[-1]) + Pmax
            def Pv2(t): return Pmax
            def Pv3(t): return Y*(t-t3[-1]) + PEEP
            def Pv4(t): return PEEP

            def dPv1_dt(t): return X
            def dPv2_dt(t): return 0
            def dPv3_dt(t): return Y
            def dPv4_dt(t): return 0

            # Defining the function for second order ODE:
            def g1(x, t): return [x[1], (R*dPv1_dt(t) + E*(Pv1(t)-PEEP) -
                                         ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

            def g2(x, t): return [x[1], (R*dPv2_dt(t) + E*(Pv2(t)-PEEP) -
                                         ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

            def g3(x, t): return [x[1], (R*dPv3_dt(t) + E*(Pv3(t)-PEEP) -
                                         ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]
            def g4(x, t): return [x[1], (R*dPv4_dt(t) + E*(Pv4(t)-PEEP) -
                                         ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

            # Solving the second order ODE for Volume and Flow
            vv1 = odeint(g1, vo, t1)
            vv2 = odeint(g2, [vv1[-1, 0], vv1[-1, 1]], t2)
            vv3 = odeint(g3, [vv2[-1, 0], vv2[-1, 1]], t3)
            vv4 = odeint(g4, [vv3[-1, 0], vv3[-1, 1]], t4)

            # Obtaining value for flow rate
            Qv1 = vv1[:, 1]
            Qv2 = vv2[:, 1]
            Qv3 = vv3[:, 1]
            Qv4 = vv4[:, 1]

            # Obtaining value for volume
            Vv1 = vv1[:, 0]
            Vv2 = vv2[:, 0]
            Vv3 = vv3[:, 0]
            Vv4 = vv4[:, 0]

            # Obtaining junction pressure
            Pi1 = Pv1(t1) - Rc*Qv1
            Pi2 = Pv2(t2)*py.ones(len(t2)) - Rc*Qv2
            Pi3 = Pv3(t3) - Rc*Qv3
            Pi4 = Pv4(t4)*py.ones(len(t4)) - Rc*Qv4

            # --------------------------------Patient 1----------------------------------------------------
            def dv_dt(P, v): return (P - PEEP - E1*v)/R1
            Vo = 0
            V1Pa1 = picardI(dv_dt, Pi1, t1, Vo, dt)
            V2Pa1 = picardI(dv_dt, Pi2, t2, V1Pa1[-1], dt)
            V3Pa1 = picardI(dv_dt, Pi3, t3, V2Pa1[-1], dt)
            V4Pa1 = picardI(dv_dt, Pi4, t4, V3Pa1[-1], dt)

            Q1Pa1 = py.gradient(V1Pa1, dt, axis=0, edge_order=2)
            Q2Pa1 = py.gradient(V2Pa1, dt, axis=0, edge_order=2)
            Q3Pa1 = py.gradient(V3Pa1, dt, axis=0, edge_order=2)
            Q4Pa1 = py.gradient(V4Pa1, dt, axis=0, edge_order=2)

            VsimP1 = py.concatenate([V1Pa1, V2Pa1, V3Pa1, V4Pa1])
            QsimP1 = py.concatenate([Q1Pa1, Q2Pa1, Q3Pa1, Q4Pa1])

            # --------------------------------Patient 2----------------------------------------------------
            def dv_dt(P, v): return (P - PEEP - E2*v)/R2
            V1Pa1 = picardI(dv_dt, Pi1, t1, Vo, dt)
            V2Pa2 = picardI(dv_dt, Pi2, t2, V1Pa2[-1], dt)
            V3Pa2 = picardI(dv_dt, Pi3, t3, V2Pa2[-1], dt)
            V4Pa2 = picardI(dv_dt, Pi4, t4, V3Pa2[-1], dt)

            Q1Pa2 = py.gradient(V1Pa2, dt, axis=0, edge_order=2)
            Q2Pa2 = py.gradient(V2Pa2, dt, axis=0, edge_order=2)
            Q3Pa2 = py.gradient(V3Pa2, dt, axis=0, edge_order=2)
            Q4Pa2 = py.gradient(V4Pa2, dt, axis=0, edge_order=2)

            VsimP2 = py.concatenate([V1Pa2, V2Pa2, V3Pa2, V4Pa2])
            QsimP2 = py.concatenate([Q1Pa2, Q2Pa2, Q3Pa2, Q4Pa2])

            # Obtaining the error of tidal volume
            VTsim1 = max(VsimP1)
            VT1_err = (abs(VTsim1 - VTact1) / VTact1)*100

            VTsim2 = max(VsimP2)
            VT2_err = (abs(VTsim2 - VTact2) / VTact2)*100

            # Checking for convergence
            dVT1 = VT1_err - VT1
            dVT2 = VT2_err - VT2

            if dVT1 and dVT2 > 0:
                Pmax = Pmax - 0.5
                print("Convergence limit has reached!")
                break

            Ppeak = max(Pi2)

            if Ppeak >= 30:
                Pmax = Pmax - 0.5
                print('Peak Pressure has exceeded safety limit!')
                break

            # Storing the value
            VT1 = VT1_err
            VT2 = VT2_err

        # Duplicating the final result with several periods
        # Reconstruct the data with the final Pmax
        DP = Pmax - PEEP
        X = DP/(t1[-1] - t1[0])
        Y = -DP/(t3[-1] - t3[0])

        def Pv1(t): return X*(t-t1[-1]) + Pmax
        def Pv2(t): return Pmax
        def Pv3(t): return Y*(t-t3[-1]) + PEEP
        def Pv4(t): return PEEP

        # Storing the values
        P = py.concatenate([Pv1(t1), py.ones(len(t2))*Pv2(t2),
                           Pv3(t3), py.ones(len(t4))*Pv4(t4)])

        def dPv1_dt(t): return X
        def dPv2_dt(t): return 0
        def dPv3_dt(t): return Y
        def dPv4_dt(t): return 0

        # Defining the function for second order ODE:(*Take note of the "PEEP" in the equation)
        def g1(x, t): return [x[1], (R*dPv1_dt(t) + E*(Pv1(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

        def g2(x, t): return [x[1], (R*dPv2_dt(t) + E*(Pv2(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

        def g3(x, t): return [x[1], (R*dPv3_dt(t) + E*(Pv3(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]
        def g4(x, t): return [x[1], (R*dPv4_dt(t) + E*(Pv4(t)-PEEP) -
                                     ((R2 + Rc)*E1 + (R1 + Rc)*E2)*x[1] - E1*E2*x[0])/(R1*R2 + Rc*R)]

        # Solving the second order ODE for Volume and Flow
        vv1 = odeint(g1, vo, t1)
        vv2 = odeint(g2, [vv1[-1, 0], vv1[-1, 1]], t2)
        vv3 = odeint(g3, [vv2[-1, 0], vv2[-1, 1]], t3)
        vv4 = odeint(g4, [vv3[-1, 0], vv3[-1, 1]], t4)

        # Obtaining value for flow rate
        Qv1 = vv1[:, 1]
        Qv2 = vv2[:, 1]
        Qv3 = vv3[:, 1]
        Qv4 = vv4[:, 1]

        # Obtaining value for volume
        Vv1 = vv1[:, 0]
        Vv2 = vv2[:, 0]
        Vv3 = vv3[:, 0]
        Vv4 = vv4[:, 0]

        # Storing the values
        Vv = py.concatenate([Vv1, Vv2, Vv3, Vv4])
        Qv = py.concatenate([Qv1, Qv2, Qv3, Qv4])

        # Obtaining junction pressure
        Pi1 = Pv1(t1) - Rc*Qv1
        Pi2 = Pv2(t2)*py.ones(len(t2)) - Rc*Qv2
        Pi3 = Pv3(t3) - Rc*Qv3
        Pi4 = Pv4(t4)*py.ones(len(t4)) - Rc*Qv4

        Pj = py.concatenate([Pi1, Pi2, Pi3, Pi4])

        # --------------------------------Patient 1----------------------------------------------------
        def dv_dt(P, v): return (P - PEEP - E1*v)/R1
        Vo = 0
        V1Pa1 = picardI(dv_dt, Pi1, t1, Vo, dt)
        V2Pa1 = picardI(dv_dt, Pi2, t2, V1Pa1[-1], dt)
        V3Pa1 = picardI(dv_dt, Pi3, t3, V2Pa1[-1], dt)
        V4Pa1 = picardI(dv_dt, Pi4, t4, V3Pa1[-1], dt)

        Q1Pa1 = py.gradient(V1Pa1, dt, axis=0, edge_order=2)
        Q2Pa1 = py.gradient(V2Pa1, dt, axis=0, edge_order=2)
        Q3Pa1 = py.gradient(V3Pa1, dt, axis=0, edge_order=2)
        Q4Pa1 = py.gradient(V4Pa1, dt, axis=0, edge_order=2)

        VsimP1 = py.concatenate([V1Pa1, V2Pa1, V3Pa1, V4Pa1])
        QsimP1 = py.concatenate([Q1Pa1, Q2Pa1, Q3Pa1, Q4Pa1])

        # --------------------------------Patient 2----------------------------------------------------
        def dv_dt(P, v): return (P - PEEP - E2*v)/R2
        V1Pa1 = picardI(dv_dt, Pi1, t1, Vo, dt)
        V2Pa2 = picardI(dv_dt, Pi2, t2, V1Pa2[-1], dt)
        V3Pa2 = picardI(dv_dt, Pi3, t3, V2Pa2[-1], dt)
        V4Pa2 = picardI(dv_dt, Pi4, t4, V3Pa2[-1], dt)

        Q1Pa2 = py.gradient(V1Pa2, dt, axis=0, edge_order=2)
        Q2Pa2 = py.gradient(V2Pa2, dt, axis=0, edge_order=2)
        Q3Pa2 = py.gradient(V3Pa2, dt, axis=0, edge_order=2)
        Q4Pa2 = py.gradient(V4Pa2, dt, axis=0, edge_order=2)

        VsimP2 = py.concatenate([V1Pa2, V2Pa2, V3Pa2, V4Pa2])
        QsimP2 = py.concatenate([Q1Pa2, Q2Pa2, Q3Pa2, Q4Pa2])

        # Tidal volume per unit weight (mL/kg)
        VTP1 = round(max(VsimP1)*1000/W1, 2)
        VTP2 = round(max(VsimP2)*1000/W2, 2)
        Err1 = round((abs(max(VsimP1) - VTact1) / VTact1)*100, 2)
        Err2 = round((abs(max(VsimP2) - VTact2) / VTact2)*100, 2)

        for i in range(1, Ti):
            # Defining time range
            t1 = py.arange(i*T, tr + T*i + dt, dt)
            t2 = py.arange(t1[-1], ti + T*i + dt, dt)
            t3 = py.arange(t2[-1], te + T*i + dt, dt)
            t4 = py.arange(t3[-1], T + T*i, dt)

            t_temp = py.concatenate([t1, t2, t3, t4])
            t = py.concatenate([t, t_temp])

            # Ventilator Setting
            Qv_temp = Qv
            Qv = py.concatenate([Qv, Qv_temp])
            Vv_temp = Vv
            Vv = py.concatenate([Vv, Vv_temp])
            P_temp = P
            P = py.concatenate([P, P_temp])

            # Junction Pressure
            Pj_temp = Pj
            Pj = py.concatenate([Pj, Pj_temp])

            # Patient 1
            QsimP1_temp = QsimP1
            QsimP1 = py.concatenate([QsimP1, QsimP1_temp])
            VsimP1_temp = VsimP1
            VsimP1 = py.concatenate([VsimP1, VsimP1_temp])

            # Patient 2
            QsimP2_temp = QsimP2
            QsimP2 = py.concatenate([QsimP2, QsimP2_temp])
            VsimP2_temp = VsimP2
            VsimP2 = py.concatenate([VsimP2, VsimP2_temp])

        # ----------------------------------------- Plotting the graph for final result-----------------------------------------
        # Creating subplots for summary tables
        figPtab = make_subplots(rows=2, cols=1,
                                row_heights=[0.3, 0.7],
                                vertical_spacing=0.05,
                                subplot_titles=(
                                    "Recommended Ventilator Setting", "Patients"),
                                specs=[[{"type": "table"}],
                                       [{"type": "table"}],
                                       ])

        # Creating subplot for MV graphs
        figPMV = make_subplots(rows=3, cols=1,
                               subplot_titles=("", "", "",
                                               "", "", ""),
                               horizontal_spacing=0.1,
                               )

        # Creating subplot for patients
        figPset = make_subplots(rows=3, cols=2,
                                subplot_titles=("Current Patient", "Pairing Patient", "",
                                                "", "", ""),
                                horizontal_spacing=0.1,
                                vertical_spacing=0.05,
                                )

        # Summarise on the MV setting
        figPtab.add_trace(go.Table(header=dict(values=['PIP (cmH2O)', 'PEEP (cmH2O)', 'RR (1/min)', 'I:E', 'Rc (cmH2Os/L)'],
                                               font=dict(size=18)
                                               ),
                                   cells=dict(values=[[round(Pmax, 1)], [PEEP], [RR], [I_E], [Rc]],
                                              font=dict(size=16),
                                              height=30
                                              )
                                   ),
                          row=1, col=1
                          )

        # Summarise on the pairing patients
        figPtab.add_trace(go.Table(header=dict(values=['', 'Current Patient ', 'Pairing Patient'],
                                               font=dict(size=18)
                                               ),
                                   cells=dict(values=[['Weight (kg)', 'Elastance (cmH2O/L)', 'Resistance (cmH2Os/L)', 'Estimated VT (mL/kg)'], [W1, E1, R1, VTP1], [W2, E2, R2, VTP2]],
                                              font=dict(size=16),
                                              height=30
                                              )
                                   ),
                          row=2, col=1
                          )

        # Plotting the graph using plotly for pressure, flow, and volume
        figPMV.add_trace(go.Scatter(x=t, y=P, name="Pressure"), row=1, col=1)
        figPMV.add_trace(go.Scatter(x=t, y=Qv, name="Flow"), row=2, col=1)
        figPMV.add_trace(go.Scatter(x=t, y=Vv, name="Volume"), row=3, col=1)

        # Patient 1
        figPset.add_trace(go.Scatter(x=t, y=Pj, name="Pressure"), row=1, col=1)
        figPset.add_trace(go.Scatter(x=t, y=QsimP1, name="Flow"), row=2, col=1)
        figPset.add_trace(go.Scatter(
            x=t, y=VsimP1, name="Volume"), row=3, col=1)
        # Patient 2
        figPset.add_trace(go.Scatter(x=t, y=Pj, name="Pressure"), row=1, col=2)
        figPset.add_trace(go.Scatter(x=t, y=QsimP2, name="Flow"), row=2, col=2)
        figPset.add_trace(go.Scatter(
            x=t, y=VsimP2, name="Volume"), row=3, col=2)

        # Update xaxis properties
        figPset.update_xaxes(title_text="<b>Time(s)<b>", title_font=dict(
            size=18), row=3, col=1)  # Similar for all graph
        figPset.update_xaxes(title_text="<b>Time(s)<b>",
                             title_font=dict(size=18), row=3, col=2)
        figPMV.update_xaxes(title_text="<b>Time(s)<b>",
                            title_font=dict(size=18), row=3, col=1)

        # Update yaxis properties
        figPset.update_yaxes(title_text="<b>Pressure (cmH2O)<b>",
                             title_font=dict(size=15), row=1, col=1)
        figPset.update_yaxes(title_text="<b>Flow (L/s)<b>",
                             title_font=dict(size=15),  row=2, col=1)
        figPset.update_yaxes(title_text="<b>Volume (L)<b>",
                             title_font=dict(size=15),  row=3, col=1)
        figPMV.update_yaxes(title_text="<b>Pressure (cmH2O)<b>",
                            title_font=dict(size=15), row=1, col=1)
        figPMV.update_yaxes(title_text="<b>Flow (L/s)<b>",
                            title_font=dict(size=15),  row=2, col=1)
        figPMV.update_yaxes(title_text="<b>Volume (L)<b>",
                            title_font=dict(size=15),  row=3, col=1)

        # Updating figure layout
        figPMV.update_layout(showlegend=False, title_text="Estimated Pressure, Flow, and Volume for Ventilator",
                             title_font=dict(size=25),
                             height=800,
                             hoverlabel=dict(
                                 font_size=18
                             )
                             )

        figPset.update_layout(showlegend=False, title_text="Estimated Pressure, Flow, and Volume for Current and Pairing Patient",
                              title_font=dict(size=25),
                              height=800,
                              hoverlabel=dict(
                                  font_size=18
                              )
                              )
        figPtab.update_annotations(font_size=18)

        divPset = figPset.to_html(full_html=False)
        divPtab = figPtab.to_html(full_html=False)
        divPMV = figPMV.to_html(full_html=False)
        return [divPtab, divPMV, divPset]
