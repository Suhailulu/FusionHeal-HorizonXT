def decide(issue: dict, candidate: dict | None, sensor_limits: dict,
           auto_accept_interpolation: bool = True,
           auto_accept_outlier_replacement: bool = False) -> dict:
    if candidate is None:
        return {"issue_id": issue["issue_id"], "candidate_id": None,
                "decision": "REVIEW", "reason": "No safe reconstruction candidate was available.",
                "confidence_value": None, "confidence_type": "not_available"}
    col = issue["sensor_id"]
    limits = sensor_limits.get(col, {})
    value = candidate["candidate_value"]
    if ("min" in limits and value < limits["min"]) or ("max" in limits and value > limits["max"]):
        return {"issue_id": issue["issue_id"], "candidate_id": candidate["candidate_id"],
                "decision": "REJECT", "reason": "Candidate violates configured range constraints.",
                "confidence_value": None, "confidence_type": "not_available"}
    if candidate["method"] == "linear_interpolation" and auto_accept_interpolation:
        return {"issue_id": issue["issue_id"], "candidate_id": candidate["candidate_id"],
                "decision": "ACCEPT", "reason": "Interior missing value interpolated and passed configured range checks.",
                "confidence_value": 0.7, "confidence_type": "heuristic"}
    if candidate["method"] == "local_median" and auto_accept_outlier_replacement:
        return {"issue_id": issue["issue_id"], "candidate_id": candidate["candidate_id"],
                "decision": "ACCEPT", "reason": "Outlier replacement enabled by configuration and passed range checks.",
                "confidence_value": 0.6, "confidence_type": "heuristic"}
    return {"issue_id": issue["issue_id"], "candidate_id": candidate["candidate_id"],
            "decision": "REVIEW", "reason": "Candidate requires human review under current policy.",
            "confidence_value": None, "confidence_type": "not_available"}
