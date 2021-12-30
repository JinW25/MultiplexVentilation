from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, FloatField
from wtforms.validators import DataRequired, Length, NumberRange


# Form for the current patient and the current ventilator setting
class PCVentilatorForm(FlaskForm):
    RR = FloatField('Respiratory Rate (1/min)', validators=[DataRequired(), NumberRange(
        min=5, max=50, message="Please enter a number within the range of 5 to 50 ")])
    PEEP = FloatField('PEEP (cmH2O)', validators=[DataRequired()])
    PIP = FloatField('PIP (cmH2O)', validators=[DataRequired()])
    I_E = FloatField('I:E', validators=[DataRequired()])
    weight = FloatField('Weight (kg)', validators=[DataRequired(), NumberRange(
        min=2,  message="Please enter a number larger than 2 ")])
    elastance = FloatField('Elastance (cmH2O/L)', validators=[DataRequired(), NumberRange(
        min=1, max=60, message="Please enter a number within the range of 1 to 60 ")])
    resistance = FloatField('Resistance (cmH2O.s/L)', validators=[DataRequired(), NumberRange(
        min=1, max=60, message="Please enter a number within the range of 1 to 60 ")])
    PatientName = StringField('Name',
                              validators=[DataRequired(), Length(min=2, max=20)])
    submit = SubmitField('Execute')

# Form for the second patient


class CandidatePatientForm(FlaskForm):
    weight2 = FloatField('Weight (kg)', validators=[DataRequired()])
    E2 = FloatField('Elastance (cmH2O/L)', validators=[DataRequired(), NumberRange(
        min=1, max=50, message="Please enter a number within the range of 1 to 50 ")])
    R2 = FloatField('Resistance (cmH2O.s/L)', validators=[DataRequired(), NumberRange(
        min=1, max=50, message="Please enter a number within the range of 1 to 50 ")])
    add = SubmitField('Add')

# Form for the final ventilator setting for co-ventilation


class PairingPatientForm(FlaskForm):
    W = FloatField('Weight (kg)', validators=[DataRequired()])
    E = FloatField('Elastance (cmH2O/L)', validators=[DataRequired(), NumberRange(
        min=1, max=50, message="Please enter a number within the range of 1 to 50 ")])
    R = FloatField('Resistance (cmH2O.s/L)', validators=[DataRequired(), NumberRange(
        min=1, max=50, message="Please enter a number within the range of 1 to 50 ")])
    Rc = FloatField('Common Resistance (cmH2O.s/L)',
                    validators=[DataRequired()])
    confirm = SubmitField('Confirm')
