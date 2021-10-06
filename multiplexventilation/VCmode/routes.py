from flask import render_template, Blueprint, Markup
from multiplexventilation.models import PatientVC
from multiplexventilation.VCmode.forms import VCVentilatorForm
from multiplexventilation.VCmode.utils import VCRampModel, VCSquareModel, VCRampPairing, VCSquarePairing


VCmode = Blueprint('VCmode', __name__)


# This route is used to generate the contour plots and the estimated graphs for the current patient
@VCmode.route("/volumecontrol", methods=['GET', 'POST'])
def volume():
    form = VCVentilatorForm()
    if form.validate_on_submit():
        if form.pattern.data == "ramp":
            mode = VCRampModel(form.weight.data, form.elastance.data, form.resistance.data, 
                        form.RR.data, form.PEEP.data, form.VT.data, form.I_E.data)
            T = 1/(form.RR.data/60)  # Respiratory Rate (Period)
            ti = T*(1-1/(1+form.I_E.data))
            PairMode = VCRampPairing(form.RR.data, form.PEEP.data, form.VT.data, form.I_E.data)
            
        elif form.pattern.data == "square":
            mode = VCSquareModel(form.weight.data,form.elastance.data, form.resistance.data, 
                        form.RR.data, form.PEEP.data, form.VT.data, form.I_E.data)
            
           
            PairMode = VCSquarePairing(form.RR.data, form.PEEP.data, form.VT.data, form.I_E.data)

        V1_graph = mode.simulate()
        Pairing_graph = PairMode.simulate()
        return render_template('volume.html', title='VC Mode', form=form, V1_graph=Markup(V1_graph), Pairing_graph=Markup(Pairing_graph))
    else:
        return render_template('volume.html', title='VC Mode', form=form, V1_graph=False, Pairing_graph=False)