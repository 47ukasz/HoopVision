# --- Importujemy potrzebne biblioteki ---
import cv2          # biblioteka do przetwarzania obrazu i wideo
import numpy as np  # biblioteka do obliczeń numerycznych (np. powierzchnia, koła, itp.)

'''
Działanie:
    1. Wczytuje plik wideo klatka po klatce.
    2. Konwertuje każdą klatkę na skalę szarości i wygładza obraz.
    3. Tworzy obraz binarny (czarno-biały) metodą progowania Otsu.
    4. Znajduje kontury potencjalnych obiektów.
    5. Dla każdego konturu oblicza:
        - pole powierzchni,
        - obwód,
        - współczynnik okrągłości (circularity).
    6. Wybiera obiekty o kształcie zbliżonym do okręgu i określonym rozmiarze.
    7. Rysuje okrąg wokół wykrytej piłki i wyświetla wynik na żywo.'''

# --- Wczytanie wideo ---
video_path = "video1.mp4"           # ścieżka do pliku wideo
cap = cv2.VideoCapture(video_path)  # otwieramy wideo do przetwarzania klatka po klatce

# --- Główna pętla przetwarzania ---
while True:
    ret, frame = cap.read()   # odczyt pojedynczej klatki
    if not ret:               # jeśli nie ma więcej klatek (koniec pliku)
        break

    # --- Przygotowanie obrazu do analizy ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # konwersja na odcienie szarości (ułatwia analizę)
    blurred = cv2.GaussianBlur(gray, (7,7), 0)      # rozmycie, by usunąć szumy i drobne detale
    
    # --- Binaryzacja obrazu (oddzielenie obiektu od tła) ---
    # cv2.THRESH_BINARY_INV odwraca kolory (piłka = biały, tło = czarne)
    # cv2.THRESH_OTSU automatycznie dobiera próg jasności
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    # --- Znajdowanie konturów na obrazie binarnym ---
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # --- Analiza każdego konturu ---
    for c in contours:
        area = cv2.contourArea(c)  # oblicz powierzchnię konturu (liczba pikseli)
        
        # pomijamy zbyt małe i zbyt duże obiekty (szumy, tło)
        if area < 100 or area > 5000:
            continue

        perimeter = cv2.arcLength(c, True)  # obwód konturu
        if perimeter == 0:
            continue

        # --- Oblicz współczynnik okrągłości (circularity) ---
        # circularity = 1 → idealne koło
        circularity = 4 * np.pi * area / (perimeter * perimeter)

        # wybieramy tylko kontury, które mają kształt zbliżony do okręgu
        if 0.6 < circularity <= 1.2:
            (x, y), radius = cv2.minEnclosingCircle(c)  # znajdź najmniejszy okrąg otaczający kontur

            # --- Dodatkowy filtr po rozmiarze ---
            # piłka ma średnicę ~50 pikseli → promień 20–35
            if 20 < radius < 35:
                center = (int(x), int(y))  # współrzędne środka
                # narysuj okrąg wokół wykrytej piłki (zielony)
                cv2.circle(frame, center, int(radius), (0,255,0), 2)
                # narysuj punkt w środku piłki (czerwony)
                cv2.circle(frame, center, 3, (0,0,255), -1)

    # --- Podgląd efektów ---
    cv2.imshow("Tracking", frame)   # główny obraz z zaznaczoną piłką
    cv2.imshow("Threshold", thresh) # podgląd binarnego obrazu (maski)

    # --- Wyjście z programu po naciśnięciu klawisza ESC ---
    if cv2.waitKey(10) & 0xFF == 27:
        break

# --- Zwolnienie zasobów po zakończeniu ---
cap.release()
cv2.destroyAllWindows()