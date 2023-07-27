class Details:
    
    ProgrammingLanguage : str = None
    Years : str = None
 
    def __init__(self, ProgrammingLanguage, Years):
        self.ProgrammingLanguage = ProgrammingLanguage
        self.Years = Years

class Request:

    Name : str = None
    Age : str = None
    Job : str = None
    Capacities : list[Details] = []

    def __init__(self, Name, Age, Job, Capacities):
        self.Name = Name
        self.Age = Age
        self.Job = Job

        for item in Capacities:
            self.Capacities.append(Request(**item))