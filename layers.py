import torch
from torchviz import make_dot
from model.eccv16 import eccv16

# Lade das Modell
colorizer_eccv16 = eccv16(pretrained=True, use_finetune_layer=True, use_training=False)
colorizer_eccv16.eval()  # Evaluation Mode

# Dummy-Eingabe für die Fine-Tuning-Layer
fine_tune_input = torch.randn(1, 313, 64, 64, requires_grad=True)  # Beispiel Input
fine_tune_layer = colorizer_eccv16.finetune_layer

# Ausgabe der Fine-Tuning-Layer berechnen
fine_tune_output = fine_tune_layer(fine_tune_input)

# Visualisierung der Fine-Tuning-Layer
fine_tune_graph = make_dot(fine_tune_output, params=dict(fine_tune_layer.named_parameters()))
fine_tune_graph.render("fine_tune_layer_architecture", format="png", cleanup=False)

print("Die Fine-Tuning-Layer wurde visualisiert und unter 'fine_tune_layer_architecture.png' gespeichert.")
