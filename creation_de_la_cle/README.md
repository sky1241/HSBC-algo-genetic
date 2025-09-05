# Création de la clé

Ce dossier contient une description succincte de la théorie des vagues d'Elliott et un script Python permettant de générer des séries de prix synthétiques respectant cette structure.

## Théorie des vagues d'Elliott

Dans son cycle de base, la théorie d'Elliott décrit huit mouvements successifs : cinq vagues motrices (impulsion) suivies de trois vagues correctives.

```
1↑ 2↓ 3↑ 4↓ 5↑ → A↓ B↑ C↓
```

### Règles essentielles

- La vague 3 n'est jamais la plus courte des vagues impulsives (1, 3, 5).
- La vague 2 ne retrace jamais sous le point de départ de la vague 1.
- La vague 4 n'empiète pas sur le territoire de la vague 1 (sauf cas de diagonales).
- La structure est fractale : chaque vague peut se décomposer en sous-vagues.

Les corrections peuvent prendre différentes formes (zigzag 5-3-5, flat 3-3-5, triangle 3-3-3-3-3, ou combinaisons complexes W-X-Y...).

