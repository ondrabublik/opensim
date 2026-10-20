"""Spustí celý výpočet test3 (kroky 1-5). Vizualizace (krok 6) se pouští zvlášť."""
import s1_model
import s2_grf
import s3_cnn
import s4_dynamika
import s5_report

if __name__ == "__main__":
    for step in (s1_model, s2_grf, s3_cnn, s4_dynamika, s5_report):
        print(f"\n===== {step.__name__} =====")
        step.main()
