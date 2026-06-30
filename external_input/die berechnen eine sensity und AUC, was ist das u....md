  
**Sensitivity** (Sensitivität) und **AUC** (Area Under the Curve) sind zwei der wichtigsten Bewertungsmetriken in der medizinischen Bildverarbeitung und Klassifikation. Gerade bei unbalancierten Datensätzen wie HAM10000 oder ISIC sagen sie weit mehr aus als die reine Genauigkeit (*Accuracy*).  
Hier ist die einfache Erklärung, was sie bedeuten und wie du sie in wenigen Zeilen Python-Code berechnest:

### **1\. Was ist Sensitivity (Sensitivität)?**

Die Sensitivität – in der Informatik auch oft **Recall** (Trefferquote) genannt  – misst, wie gut dein Modell **tatsächlich kranke Fälle (positive Fälle)** als solche erkennt.

* **Bedeutung im Paper:** Wenn 100 Patienten wirklich Hautkrebs haben, und dein Modell erkennt 92 davon richtig, liegt die Sensitivität bei 92 %. Die restlichen 8 % sind *False Negatives* (fälschlicherweise als gesund eingestufte Krebspatienten), was in der Medizin fatal sein kann.  
* **Formel:**  
  $$\\text{Sensitivity} \= \\frac{\\text{True Positives (TP)}}{\\text{True Positives (TP)} \+ \\text{False Negatives (FN)}}$$

#### **Wie berechne ich Sensitivity?**

Da du ohnehin scikit-learn benutzt, kannst du die Sensitivität direkt über die Funktion recall\_score (oder im classification\_report) ausgeben lassen:

Python  
from sklearn.metrics import recall\_score

\# y\_true: Echte Labels (z. B. 0 für gesund, 1 für Krebs)  
\# y\_pred: Die Vorhersagen deines Modells

\# Für binäre Klassifikation (z.B. ISIC 2020: Benign vs. Malignant):  
sensitivity \= recall\_score(y\_true, y\_pred)  
print(f"Sensitivity: {sensitivity:.4f}")

\# Für Multiklassen-Klassifikation (z.B. HAM10000 mit 7 Klassen):  
\# 'macro' berechnet den Durchschnitt der Sensitivitäten aller einzelnen Klassen  
sensitivity\_macro \= recall\_score(y\_true, y\_pred, average='macro')  
print(f"Macro Sensitivity: {sensitivity\_macro:.4f}")

### **2\. Was ist die AUC (Area Under the ROC Curve)?**

Die **AUC** beschreibt die Fläche unter der sogenannten *ROC-Kurve*. Sie misst die **Trennoschärfe des Modells**, also wie gut das Modell generell in der Lage ist, zwischen gesunden und kranken Bildern zu unterscheiden – und zwar unabhängig von einem festen Schwellenwert (Threshold).

* **Die Skala:**  
  * **1.0:** Das perfekte Modell. Es sortiert jede Krebs-Zelle und jedes Muttermal fehlerfrei.  
  * **0.5:** Das Modell rät rein zufällig (wie ein Münzwurf).  
* **Vorteil bei Hautkrebs:** Da die Datensätze extrem unbalanciert sind (z. B. ISIC 2020 hat über 32.000 gesunde Bilder, aber nur knapp 600 Krebsbilder), würde ein Modell, das *immer* "gesund" rät, eine Accuracy von 98% haben. Die AUC würde hier jedoch sofort entlarven, dass das Modell unbrauchbar ist und bei \~0.5 liegt. Das Paper nutzt die AUC genau aus diesem Grund, um die Leistung pro Klasse abzusichern.

#### **Wie berechne ich die AUC?**

Wichtig ist: Für die AUC benötigt man nicht die harten Vorhersagen (0 oder 1), sondern die **Wahrscheinlichkeiten (Probabilities)**, die dein Modell ausgibt (z.B. den Output nach der *Softmax*\- oder *Sigmoid*\-Funktion).

Python  
from sklearn.metrics import roc\_auc\_score

\# y\_true: Echte Klassen-Labels  
\# y\_probs: Die vom Modell vorhergesagten Wahrscheinlichkeiten für die Klassen

\# Für binäre Klassifikation (Wahrscheinlichkeiten der positiven Klasse):  
auc \= roc\_auc\_score(y\_true, y\_probs)  
print(f"AUC: {auc:.4f}")

\# Für Multiklassen-Klassifikation (z.B. HAM10000):  
\# y\_probs muss hier eine Matrix mit den Wahrscheinlichkeiten für alle 7 Klassen sein  
auc\_multi \= roc\_auc\_score(y\_true, y\_probs, multi\_class='ovr', average='macro')  
print(f"Multiclass AUC: {auc\_multi:.4f}")

### **Zusammenfassung für deinen Code:**

Am Ende deines Test-Skripts sammelst du einfach zwei Listen:

1. all\_preds: Die finalen Klassen-Entscheidungen (für die Accuracy und die Sensitivity).  
2. all\_probs: Die ungerundeten Wahrscheinlichkeiten direkt aus dem Netzwerk-Output (für die AUC).

Mit den oben stehenden Zeilen hast du innerhalb von Sekunden exakt dieselben Validierungsmetriken berechnet, die auch im Paper verwendet wurden\!