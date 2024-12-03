from torch_geometric.explain import Explainer, GNNExplainer

def explain_model(model, data, node_index=10):
    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=200),
        explanation_type='model',
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='node',
            return_type='log_probs',
        ),
    )
    explanation = explainer(data.x, data.edge_index, index=node_index)
    print(f'Generated explanations in {explanation.available_explanations}')

    feature_importance_path = 'feature_importance.png'
    explanation.visualize_feature_importance(feature_importance_path, top_k=10)
    print(f"Feature importance plot has been saved to '{feature_importance_path}'")

    subgraph_path = 'subgraph.pdf'
    explanation.visualize_graph(subgraph_path)
    print(f"Subgraph visualization plot has been saved to '{subgraph_path}'")

    return explanation

