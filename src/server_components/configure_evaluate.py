from flwr.common import EvaluateIns


def configure_evaluate_federated_learning(evaluate_ins, clients):
    return [(client, evaluate_ins) for client in clients]


def configure_evaluate_federated_unlearning(
    custom_fed_avg_instance, evaluate_ins, clients, config
):
    # Return client/config pairs
    client_evaluate_pairs = []
    for client in clients:
        if (
            custom_fed_avg_instance.command
            == "global_model_restoration_and_degraded_model_unlearning"
        ):
            client_evaluate_pairs.append(
                (
                    client,
                    EvaluateIns(
                        custom_fed_avg_instance.global_model_parameters, config
                    ),
                )
            )
        else:
            client_evaluate_pairs.append((client, evaluate_ins))

    # Return client/config pairs
    return client_evaluate_pairs
