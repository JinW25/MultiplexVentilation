from multiplexventilation import db

class PatientPC(db.Model):
    # Patient's information
    id = db.Column(db.Integer, primary_key=True) # Unique id for patient
    PatientName = db.Column(db.String(20), unique=True, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    elastance = db.Column(db.Float, nullable=False)
    resistance = db.Column(db.Float, nullable=False)

    # The corresponding ventilation setting
    RR = db.Column(db.Float, nullable=False)
    PEEP = db.Column(db.Float, nullable=False)
    PIP = db.Column(db.Float, nullable=False)
    I_E = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"PatientPC('{self.PatientName}', '{self.weight}', '{self.elastance}', '{self.resistance}', '{self.RR}', '{self.PEEP}', '{self.PIP}', '{self.I_E}')"

class PatientVC(db.Model):
    # Patient's information
    id = db.Column(db.Integer, primary_key=True) # Unique id for patient
    PatientName = db.Column(db.String(20), unique=True, nullable=False)
    weight = db.Column(db.Float, nullable=False)
    elastance = db.Column(db.Float, nullable=False)
    resistance = db.Column(db.Float, nullable=False)

    # The corresponding ventilation setting
    RR = db.Column(db.Float, nullable=False)
    PEEP = db.Column(db.Float, nullable=False)
    VT = db.Column(db.Float, nullable=False)
    I_E = db.Column(db.Float, nullable=False)
    Qin = db.Column(db.Float, nullable=False)

    def __repr__(self):
        return f"PatientVC('{self.PatientName}', '{self.weight}', '{self.elastance}', '{self.resistance}', '{self.RR}', '{self.PEEP}', '{self.VT}', '{self.I_E}', '{self.Qin}')"

class CandidatePatient(db.Model):
    # Information of patient candidates
    id = db.Column(db.Integer, primary_key=True, autoincrement=True) # Unique id for patient
    weight = db.Column(db.Float, nullable=False)
    elastance = db.Column(db.Float, nullable=False)
    resistance = db.Column(db.Float, nullable=False)
    
    def __repr__(self):
        return f"CandidatePatient('{self.weight}', '{self.elastance}', '{self.resistance}' )"