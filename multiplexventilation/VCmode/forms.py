from flask_wtf import FlaskForm
from wtforms import StringField, SubmitField, SelectField, FloatField
from wtforms.validators import DataRequired, Length, NumberRange

# Forms for ventilator setting in VC mode and current patient
class VCVentilatorForm(FlaskForm):
    RR = FloatField('Respiratory Rate (1/min)', validators=[DataRequired(), NumberRange(min=5, max=25, message="Please enter a number within the range of 5 to 25 ")])
    PEEP = FloatField('PEEP (cmH2O)', validators=[DataRequired()])
    I_E = FloatField('I:E', validators=[DataRequired()])
    weight = FloatField('Weight (kg)', validators=[DataRequired(), NumberRange(min=2,  message="Please enter a number larger than 2 ")])
    VT = FloatField('Tidal Volume (mL)', validators=[DataRequired()])
    elastance = FloatField('Elastance (cmH2O/L)', validators=[DataRequired(), NumberRange(min=1, max=50, message="Please enter a number within the range of 1 to 50 ")])
    resistance = FloatField('Resistance (cmH2O.s/L)', validators=[DataRequired(), NumberRange(min=1, max=50, message="Please enter a number within the range of 1 to 50 ")])
    PatientName = StringField('Name',
                           validators=[DataRequired(), Length(min=2, max=20)])
    submit = SubmitField('Execute')
    pattern = SelectField('Flow Wave', choices=[('square','Square'), ('ramp','Ramp')])
    
