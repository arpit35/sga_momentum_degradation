from flwr.common import FitIns


def configure_fit_federated_learning(fit_ins, clients):
    return [(client, fit_ins) for client in clients]


def configure_fit_federated_unlearning(
    custom_fed_avg_instance, fit_ins, clients, config
):
    # Return client/config pairs
    client_fit_pairs = []
    for client in clients:
        if (
            custom_fed_avg_instance.command
            == "global_model_restoration_and_degraded_model_unlearning"
            and custom_fed_avg_instance.unlearn_client_id == client.cid
        ):
            client_fit_pairs.append(
                (
                    client,
                    FitIns(custom_fed_avg_instance.degraded_model_parameters, config),
                )
            )
        elif custom_fed_avg_instance.command == "global_model_restoration":
            client_fit_pairs.append(
                (
                    client,
                    FitIns(custom_fed_avg_instance.global_model_parameters, config),
                )
            )
        else:
            client_fit_pairs.append((client, fit_ins))

    # Return client/config pairs
    return client_fit_pairs
