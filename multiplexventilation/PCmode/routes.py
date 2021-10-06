from flask import render_template, Blueprint, Markup, flash, redirect, url_for
from multiplexventilation.models import PatientPC, CandidatePatient
from multiplexventilation.PCmode.forms import PCVentilatorForm, CandidatePatientForm, PairingPatientForm
from multiplexventilation.PCmode.utils import PCModel, PCPairing, PCSetting
from multiplexventilation import db


PCmode = Blueprint('PCmode', __name__)

# This route is used to get user inputs from the form, and save the data into a database for the use of double compartment model
@PCmode.route("/pressurecontrol", methods=['GET', 'POST'])
def pressure():
    form = PCVentilatorForm()
    candidateform = CandidatePatientForm()
    candidates_data = CandidatePatient.query.all() # Getting the data of the candidates from the database
    pairingform = PairingPatientForm()
    setting_data = PatientPC.query.get(1) # Getting the previous data of the MV setting to be overwritten by the new data
    
    if form.validate_on_submit():
        # Creating empty array to store data
        weight2 = []
        E2 = []
        R2 = []

        # If the table for pairing patients consist of data
        if candidates_data:
            for i in candidates_data:
                weight2.append(i.weight)
                E2.append(i.elastance)
                R2.append(i.resistance)
        # if the table for pairing patients is empty, it will return the contour plot for the current patient
        else:
            weight2.append(form.weight.data)
            E2.append(form.elastance.data)
            R2.append(form.resistance.data)
        
        # Updating database with new data (The data will be replaced everytime the user clicked on execute)
        setting_data.PIP = form.PIP.data
        setting_data.PEEP = form.PEEP.data
        setting_data.RR = form.RR.data
        setting_data.I_E = form.I_E.data
        setting_data.elastance = form.elastance.data
        setting_data.resistance = form.resistance.data
        setting_data.weight = form.weight.data
        setting_data.PatientName = form.PatientName.data
        db.session.commit()

        # Calling the function from utils.py to obtain the estimated graph for current patient
        mode = PCModel(form.weight.data, form.elastance.data, form.resistance.data, 
                    form.RR.data, form.PEEP.data, form.PIP.data, form.I_E.data)

        # Calling the function from utils.py to generate the contour plots
        PairMode = PCPairing(weight2, E2, R2,
                            form.RR.data, form.PEEP.data, form.PIP.data, form.I_E.data)

        # Simulating
        P1_graph = mode.simulate()
        Pairing_graph = PairMode.simulate()

        # Printing the graph on to the specified div in html
        return render_template('pressure.html', title='PC Mode', form=form, candidateform=candidateform, 
                                P1_graph=Markup(P1_graph), Pairing_graph=Markup(Pairing_graph), candidates=candidates_data, pairingform = pairingform) 
    else:

        # If no simulations are done, no grarphs will be generated
        return render_template('pressure.html', title='PC Mode', form=form, candidateform=candidateform, candidates=candidates_data, 
                                P1_graph=False, Pairing_graph="",
                                pairingform = pairingform) 
    
# This route is used to delete the candidate from the list of pairing patients
@PCmode.route("/pressurecontrol/<int:row_id>/delete", methods=['GET', 'POST'])
def delete_candidate(row_id):
    row = CandidatePatient.query.get(row_id)
    db.session.delete(row)
    db.session.commit()
    flash('The patient has been deleted!', 'danger')
    return redirect(url_for('PCmode.pressure'))

# This route is used to add candidate to the list of pairing patients
@PCmode.route("/addpatient", methods=['POST'])
def add_candidate():
    candidateform = CandidatePatientForm()
    if candidateform.validate_on_submit():
        candidate = CandidatePatient(weight=candidateform.weight2.data, elastance =candidateform.E2.data, resistance =candidateform.R2.data)
        db.session.add(candidate)
        db.session.commit()
        flash('Patient added successfully!', 'success')
        return redirect(url_for('PCmode.pressure'))
    return redirect(url_for('PCmode.pressure'))
    
# This route calls the function for utils.py to simulate the actual ventilator setting using double compartment model
@PCmode.route("/pairpatient", methods=['GET', 'POST'])
def pair_patient():
    pairingform = PairingPatientForm()
    setting_data = PatientPC.query.get(1)

    if pairingform.validate_on_submit():
        pairpatient = PCSetting(setting_data.weight, pairingform.W.data, setting_data.elastance, setting_data.resistance, 
                                pairingform.E.data, pairingform.R.data, pairingform.Rc.data, setting_data.RR, 
                                setting_data.PEEP, setting_data.PIP, setting_data.I_E)
        pairResult = pairpatient.simulate()
        return render_template("pairPatient.html", title = 'Pairing Results', Ptab=Markup(pairResult[0]), PMV=Markup(pairResult[1]), Pset=Markup(pairResult[2]))
    return render_template("pairPatient.html", title = 'Pairing Results', Ptab=False)

    