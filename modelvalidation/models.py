from re import T
from django.db import models

# Create your models here.


class CSVData:
    months_as_customer: str
    age: str
    policy_number: str
    policy_bind_date: str
    policy_state: str
    policy_csl: str
    policy_deductable:   str
    policy_annual_premium:   float
    umbrella_limit:      str
    insured_zip: str
    insured_sex: str
    insured_education_level:   str
    insured_occupation:      str
    insured_hobbies:     str
    insured_relationship:    str
    capital_gains:       str
    capital_loss: str
    incident_date:       str
    incident_type:      str
    collision_type:    str
    incident_severity:   str
    authorities_contacted:   str
    incident_state:      str
    incident_city:       str
    incident_location:   str
    incident_hour_of_the_day:    str
    number_of_vehicles_involved: str
    property_damage: str
    bodily_injuries:     str
    witnesses: str
    police_report_available: str
    total_claim_amount:      str
    injury_claim: str
    property_claim:      str
    vehicle_claim:      str
    auto_make: str
    auto_model: str
    auto_year: str
    fraud_reported: str


class descData:
    colName: str
    count_val: str
    mean_val: str
    std_val: str
    min_val: str
    per25_val: str
    per50_val: str
    per75_val: str
    max_val: str


class missingDataList:
    colName: str
    dtType: str
    count_rows: int
    total_rows: int
    missing_rows: int


class lstColFreq:
    colName: str
    freqVal: dict
    total_rows: int
    missing_rows: int


class lstOutlierGrubbs:
    colName: str
    min_location: str
    max_location: str
    min_value: str
    max_value: str


class lstOutlieranomalies:
    colName: str
    lower_limit: str
    upper_limit: str
    arr_anomalies: str


class lstTestModelPerf:
    testName: str
    testResult: str
    testResult_dict: dict


class lstCnfrmSrc:
    colId: str
    colName: str
    srcName: str
    emailId: str
    reqResp: str
    dataQlt: str
    comment: str
