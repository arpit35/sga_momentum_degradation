def aggregate_evaluate_federated_unlearning(custom_fed_avg_instance):
    if custom_fed_avg_instance.command == "global_model_unlearning":
        custom_fed_avg_instance.command = (
            "global_model_restoration_and_degraded_model_unlearning"
        )

    elif (
        custom_fed_avg_instance.command
        == "global_model_restoration_and_degraded_model_unlearning"
    ):
        if (
            custom_fed_avg_instance.current_knowledge_eraser_round
            == custom_fed_avg_instance.knowledge_eraser_rounds
        ):
            custom_fed_avg_instance.command = "global_model_restoration"
        else:
            custom_fed_avg_instance.current_knowledge_eraser_round += 1
            custom_fed_avg_instance.command = "degraded_model_refinement"
