# LAB — Sauvegarde chiffrée automatisée d’une base de données avec HashiCorp Vault, S3 et restauration contrôlée

## Contexte

Dans ce lab, vous devez concevoir une solution complète de sauvegarde sécurisée et de restauration d’une base de données.

L’objectif est de mettre en place une chaîne de traitement fiable permettant :

* de récupérer les secrets nécessaires à la connexion à la base ;
* de gérer une clé symétrique avec HashiCorp Vault ;
* de produire un dump de base de données ;
* de calculer un hash avant chiffrement ;
* de chiffrer le dump ;
* de calculer un hash après chiffrement ;
* de générer un fichier JSON de métadonnées accompagnant l’archive ;
* d’envoyer l’archive chiffrée et son JSON dans un bucket S3 ;
* de supprimer le dump en clair du disque local ;
* puis de restaurer une archive choisie par l’utilisateur avec plusieurs contrôles d’intégrité.

Une fois ces deux scripts validés, vous devrez automatiser la sauvegarde via un cron job exécuté toutes les 2 minutes.

---

# Objectif général

Mettre en place une solution de sauvegarde et de restauration sécurisée fondée sur :

* Python ;
* une base de données relationnelle ;
* HashiCorp Vault pour les secrets et les clés ;
* Amazon S3 pour l’archivage ;
* cron pour l’automatisation.

---

# Résultat attendu

À la fin du lab, vous devrez disposer de :

* un script Python de sauvegarde chiffrée ;
* un script Python de restauration ;
* un bucket S3 contenant les archives chiffrées et leurs fichiers JSON associés ;
* une automatisation cron exécutant la sauvegarde toutes les 2 minutes.

---

# Partie 1 — Script de sauvegarde chiffrée

## Nom attendu du script

`backup_secure.py`

## Rôle du script

Ce script doit produire une sauvegarde chiffrée d’une base de données, calculer les empreintes d’intégrité, générer les métadonnées associées, envoyer le tout dans un bucket S3, puis supprimer toute trace locale du dump en clair.

---

## Fonctionnement attendu

### 1. Récupération des informations sensibles

Le script doit récupérer les éléments nécessaires à la connexion à la base de données.

Deux approches sont acceptées :

#### Approche A

Le mot de passe de la base de données est stocké dans HashiCorp Vault et récupéré dynamiquement par le script.

#### Approche B

Le mot de passe de la base est récupéré par un autre mécanisme sécurisé, mais la gestion de la clé symétrique doit obligatoirement impliquer HashiCorp Vault.

Dans tous les cas, aucun secret ne doit être écrit en dur dans le code source.

---

### 2. Gestion de la clé symétrique

Le script doit utiliser une clé symétrique pour chiffrer le dump.

Deux stratégies sont autorisées :

#### Option 1

Le script génère une clé symétrique en Python, puis l’enregistre dans HashiCorp Vault.

#### Option 2

La clé symétrique est générée directement via HashiCorp Vault, puis récupérée par le script au moment du chiffrement.

Le choix retenu devra être expliqué dans la documentation.

---

### 3. Génération du dump de la base de données

Le script doit produire un dump complet de la base dans un fichier temporaire local.

Contraintes :

* le fichier doit être exploitable pour une restauration complète ;
* son nom doit contenir la date et l’heure ;
* le dump en clair ne doit exister que le temps strictement nécessaire.

Exemple de nommage :

`db_backup_YYYYMMDD_HHMMSS.sql`

---

### 4. Calcul du hash avant chiffrement

Avant le chiffrement, le script doit calculer une empreinte cryptographique du dump en clair.

Cette empreinte doit permettre d’identifier exactement le contenu d’origine.

Le fichier JSON devra contenir au minimum :

* l’algorithme de hash utilisé ;
* le hash du dump avant chiffrement ;
* la taille du dump en clair.

---

### 5. Chiffrement du dump

Le dump doit ensuite être chiffré à l’aide de la clé symétrique obtenue précédemment.

Contraintes :

* le fichier chiffré doit être exploitable ultérieurement pour une restauration ;
* son nom doit être cohérent avec celui du dump ;
* le mécanisme de chiffrement retenu doit être documenté.

Exemple de nommage :

`db_backup_YYYYMMDD_HHMMSS.enc`

---

### 6. Calcul du hash après chiffrement

Une fois l’archive chiffrée produite, le script doit calculer une seconde empreinte cryptographique.

Cette empreinte permettra :

* de vérifier l’intégrité du fichier chiffré stocké ;
* de détecter toute altération ou corruption après archivage.

---

### 7. Génération du fichier JSON de métadonnées

Pour chaque archive chiffrée, le script doit créer un fichier JSON associé.

Ce fichier JSON doit accompagner systématiquement l’archive chiffrée dans le bucket S3, au même emplacement.

Il doit contenir au minimum :

* la date de création ;
* l’heure de création ;
* le nom du dump en clair ;
* le nom du fichier chiffré ;
* le type de base de données ;
* le nom de la base ;
* l’algorithme de hash utilisé ;
* le hash avant chiffrement ;
* le hash après chiffrement ;
* l’algorithme de chiffrement utilisé ;
* la taille du dump en clair ;
* la taille du fichier chiffré ;
* la référence de la clé dans Vault ;
* le statut de l’opération ;
* un message d’erreur si nécessaire.

Exemple de nommage :

`db_backup_YYYYMMDD_HHMMSS.json`

---

### 8. Envoi vers le bucket S3

Le script doit envoyer dans le bucket S3 :

* le fichier chiffré ;
* le fichier JSON correspondant.

Contraintes :

* les deux fichiers doivent être stockés côte à côte ;
* l’organisation du bucket doit permettre d’identifier facilement les sauvegardes ;
* le chemin doit être cohérent et exploitable pour la restauration.

Exemple de structure possible :

`backups/2026/04/17/db_backup_20260417_103000.enc`
`backups/2026/04/17/db_backup_20260417_103000.json`

---

### 9. Suppression du dump en clair

Une fois l’envoi vers S3 terminé avec succès, le script doit supprimer le dump SQL non chiffré présent localement.

Cette étape est obligatoire.

Aucun dump en clair ne doit rester sur le disque après exécution normale du script.

---

### 10. Journalisation

Le script doit produire des logs suffisamment détaillés pour permettre le suivi des opérations suivantes :

* récupération des secrets ;
* génération ou récupération de la clé ;
* création du dump ;
* calcul du hash avant chiffrement ;
* chiffrement ;
* calcul du hash après chiffrement ;
* génération du JSON ;
* envoi dans S3 ;
* suppression du dump en clair ;
* erreurs éventuelles.

---

# Partie 2 — Script de restauration contrôlée

## Nom attendu du script

`restore_secure.py`

## Rôle du script

Ce script doit permettre d’analyser les sauvegardes présentes dans le bucket S3, de proposer un choix à l’utilisateur, de restaurer l’archive sélectionnée dans la base de données, puis d’effectuer plusieurs vérifications successives d’intégrité.

---

## Fonctionnement attendu

### 1. Analyse du bucket S3

Le script doit parcourir le bucket S3 afin de détecter les sauvegardes disponibles.

Il doit être capable de retrouver :

* les fichiers chiffrés ;
* les fichiers JSON associés.

---

### 2. Listing des archives disponibles

Le script doit afficher à l’utilisateur une liste lisible des sauvegardes disponibles.

Chaque entrée doit contenir au minimum :

* un identifiant de sélection ;
* le nom de l’archive ;
* la date ;
* l’heure ;
* la taille ;
* le statut ou toute information utile issue du JSON.

Exemple attendu :

* Sauvegarde 1 — 2026-04-17 10:30:00
* Sauvegarde 2 — 2026-04-17 10:32:00
* Sauvegarde 3 — 2026-04-17 10:34:00

---

### 3. Choix d’une archive par l’utilisateur

Le script doit permettre à l’utilisateur de choisir l’archive à restaurer.

Le choix peut se faire :

* par numéro ;
* par nom de fichier ;
* ou par date et heure.

Le mécanisme retenu doit rester simple et clair.

---

### 4. Téléchargement de l’archive et du JSON associé

Après sélection, le script doit télécharger :

* l’archive chiffrée ;
* le fichier JSON correspondant.

---

### 5. Vérification 1 — Contrôle de l’archive chiffrée

Avant tout déchiffrement, le script doit recalculer le hash du fichier chiffré téléchargé.

Ce hash doit être comparé avec le hash après chiffrement stocké dans le fichier JSON.

Objectif :

* vérifier que l’archive stockée dans S3 n’a subi aucune modification ;
* confirmer que le fichier téléchargé est strictement identique à celui qui a été envoyé lors de la sauvegarde.

Si les deux valeurs ne correspondent pas, la restauration doit être interrompue immédiatement.

---

### 6. Récupération de la clé symétrique

Si l’archive est valide, le script doit récupérer dans HashiCorp Vault la clé symétrique nécessaire au déchiffrement.

Cette récupération doit s’appuyer sur la référence enregistrée dans le fichier JSON ou sur votre logique d’implémentation documentée.

---

### 7. Déchiffrement de l’archive

Le script doit déchiffrer l’archive afin de reconstituer le dump SQL d’origine.

---

### 8. Vérification 2 — Contrôle du dump déchiffré

Une fois le dump reconstitué, le script doit recalculer son hash.

Ce hash doit être comparé avec le hash avant chiffrement présent dans le fichier JSON.

Objectif :

* vérifier que le déchiffrement a bien restitué exactement le dump initial ;
* confirmer que la clé utilisée et le processus de déchiffrement sont corrects.

Si les deux valeurs ne correspondent pas, la restauration doit être interrompue.

---

### 9. Restauration dans la base de données

Si les contrôles précédents sont conformes, le script doit restaurer directement le dump dans la base cible.

Contraintes :

* la restauration doit être entièrement pilotée par le script ;
* les opérations doivent être journalisées ;
* les erreurs doivent être clairement affichées.

---

### 10. Vérification 3 — Contrôle après restauration

Une fois la restauration terminée, le script doit effectuer un contrôle final.

Pour cela, il doit :

* produire un nouveau dump de contrôle de la base restaurée ;
* calculer le hash de ce nouveau dump ;
* comparer cette valeur avec le hash avant chiffrement du dump d’origine, stocké dans le JSON.

Objectif :

* vérifier que la base restaurée correspond réellement à la sauvegarde initiale ;
* s’assurer que le résultat final en base est cohérent avec le dump sauvegardé.

Si les valeurs ne correspondent pas, la restauration doit être considérée comme invalide ou incomplète.

---

### 11. Nettoyage local

À la fin de l’opération, le script doit supprimer :

* le fichier chiffré téléchargé localement ;
* le dump temporaire déchiffré ;
* le dump de contrôle éventuellement généré après restauration.

Aucun fichier sensible ne doit rester sur le disque à la fin de l’exécution.

---

# Partie 3 — Automatisation par cron

Une fois les deux scripts terminés et validés, vous devez automatiser l’exécution du script de sauvegarde.

## Exigence

Le script `backup_secure.py` doit être lancé automatiquement toutes les 2 minutes via cron.

## Résultat attendu

Vous devrez fournir :

* la ligne cron exacte ;
* la preuve que le script s’exécute correctement ;
* les logs associés ;
* plusieurs archives consécutives visibles dans le bucket S3.

---

# Contraintes techniques

## Langage

* Python

## Composants à utiliser

* base de données relationnelle ;
* HashiCorp Vault ;
* bucket S3 ;
* cron sous Linux.

---

# Contraintes de sécurité

La solution devra impérativement respecter les règles suivantes :

* aucun mot de passe ne doit être écrit en dur dans le code ;
* aucune clé ne doit être écrite en dur dans le code ;
* le dump en clair doit être supprimé après traitement ;
* l’intégrité doit être vérifiée avant chiffrement, après chiffrement, avant restauration, après déchiffrement et après restauration ;
* les erreurs doivent être gérées proprement ;
* les logs doivent être clairs et exploitables.

---

# Livrables attendus

Vous devez rendre les éléments suivants.

## 1. Scripts Python

* `backup_secure.py`
* `restore_secure.py`

## 2. Configuration

* un fichier de configuration d’exemple si nécessaire ;
* les variables d’environnement utiles ;
* les instructions d’installation et d’exécution.

## 3. Contenu S3

Une démonstration montrant clairement :

* les archives chiffrées ;
* les fichiers JSON associés ;
* plusieurs sauvegardes horodatées.

## 4. Automatisation

* la ligne cron exacte ;
* une explication de son fonctionnement ;
* les traces d’exécution.

## 5. Documentation technique

Cette documentation devra expliquer :

* l’architecture générale ;
* le rôle de HashiCorp Vault ;
* le rôle de S3 ;
* la logique de sauvegarde ;
* la logique de restauration ;
* la stratégie de gestion des clés ;
* le mécanisme de hash ;
* le mécanisme de chiffrement ;
* les limites éventuelles de la solution.

---

# Critères d’évaluation

## Fonctionnalité

* le dump est bien généré ;
* le dump est bien chiffré ;
* le JSON est correctement généré ;
* l’archive et le JSON sont correctement envoyés dans S3 ;
* le dump en clair est supprimé ;
* la restauration fonctionne ;
* l’utilisateur peut choisir une archive.

## Sécurité

* les secrets sont correctement gérés ;
* la clé symétrique est correctement traitée avec Vault ;
* les contrôles d’intégrité sont réalisés à chaque étape critique ;
* le nettoyage local est effectif.

## Robustesse

* les erreurs sont gérées ;
* les logs sont lisibles ;
* le script continue proprement ou s’arrête correctement selon le cas.

## Exploitabilité

* les noms de fichiers sont cohérents ;
* les métadonnées JSON sont complètes ;
* le listing des sauvegardes est compréhensible ;
* le cron fonctionne réellement.

---

# Déroulement attendu lors de la démonstration

## Démonstration de la sauvegarde

Vous devrez montrer :

1. la récupération du mot de passe de base ;
2. la génération ou la récupération de la clé symétrique ;
3. la création du dump ;
4. le calcul du hash avant chiffrement ;
5. le chiffrement ;
6. le calcul du hash après chiffrement ;
7. la création du fichier JSON ;
8. l’envoi des fichiers dans S3 ;
9. la suppression du dump en clair.

## Démonstration de la restauration

Vous devrez montrer :

1. l’analyse du bucket ;
2. le listing des sauvegardes disponibles ;
3. le choix d’une archive ;
4. le téléchargement ;
5. la vérification du hash de l’archive chiffrée ;
6. le déchiffrement ;
7. la vérification du hash du dump déchiffré ;
8. la restauration en base ;
9. la génération du dump de contrôle ;
10. la vérification finale du hash après restauration ;
11. le nettoyage local.

## Démonstration de l’automatisation

Vous devrez montrer :

1. la configuration du cron ;
2. l’exécution automatique toutes les 2 minutes ;
3. la présence de plusieurs archives successives dans S3.

---

# Bonus possibles

Des points bonus peuvent être attribués si vous ajoutez :

* une compression avant chiffrement ;
* une politique de rétention ;
* un versionnement des sauvegardes ;
* une restauration vers une base de test ;
* un menu interactif plus propre ;
* une notification en cas d’échec ;
* une architecture de code séparant configuration, logique métier et utilitaires.

---

# Consigne importante

Ce lab doit être réalisé uniquement dans un environnement de test maîtrisé et autorisé.

Toutes les manipulations doivent être effectuées sur :

* une base de données dont vous avez le contrôle ;
* un bucket S3 vous appartenant ou prévu pour le lab ;
* une instance HashiCorp Vault de test ou de démonstration.

---

# Résumé fonctionnel attendu

## Script 1 — Sauvegarde

Le script doit :

* récupérer les secrets ;
* obtenir la clé symétrique ;
* créer un dump ;
* calculer le hash du dump ;
* chiffrer le dump ;
* calculer le hash du fichier chiffré ;
* générer un JSON de métadonnées ;
* envoyer archive et JSON dans S3 ;
* supprimer le dump en clair.

## Script 2 — Restauration

Le script doit :

* analyser le bucket ;
* lister les archives ;
* laisser l’utilisateur en choisir une ;
* télécharger archive et JSON ;
* vérifier l’intégrité du fichier chiffré ;
* récupérer la clé ;
* déchiffrer ;
* vérifier l’intégrité du dump reconstitué ;
* restaurer la base ;
* refaire un dump de contrôle ;
* comparer le hash final avec le hash d’origine ;
* nettoyer les fichiers temporaires.

## Automatisation

Le cron doit exécuter la sauvegarde toutes les 2 minutes.


