(defgroup -faces nil "Faces for -mode")

(defface -keyword-face
    `((t (:foreground "DodgerBlue2")))
  "Face used to keywords, like `load'"
  :group '-faces)

(defface -assigned-identifier-face
    `((t (:foreground "medium purple")))
  "Face used to highlight identifiers being assigned to with a colon"
  :group '-faces)

(defface -symbol-verb-face
    `((t (:foreground "HotPink2")))
  "Face used to highlight verbs with symbolic names"
  :group '-faces)

(defface -identifier-verb-face
    `((t (:foreground "plum3")))
  "Face used to highlight verbs with identifier names"
  :group '-faces)

(defface -adverb-face
    `((t (:foreground "medium orchid")))
  "Face used to highlight primitive adverbs"
  :group '-faces)

(defface -punctuation-face
    `((t (:foreground "DodgerBlue2")))
  "Face used to highlight punctuation, like ; and ->"
  :group '-faces)

(defconst -keywords
  (regexp-opt (split-string "load") 'words)
  "Keywords")

(defconst -syntax-table
  (let ((table (make-syntax-table)))
    (modify-syntax-entry ?_ "w" table)
    (modify-syntax-entry ?$ "." table)
    (modify-syntax-entry ?% "." table)
    table))

(defconst -symbol-verbs
  (regexp-opt (split-string "|| && == =! <: >: #: // <= >= ?:
                             - @ , + * # / ^ | ! $ = < > % ? &")))

(defconst -symbol-adverbs
  (regexp-opt (split-string "`: | . ' ` ~ \\")))

(setq -mode-fontlock
      `(;; Need OVERRIDE=t or else comments containing string literals won't
        ;; highlight properly.
        (,-keywords . '-punctuation-face)
        ("\\\\ .*" . (0 'font-lock-comment-face t))  
        ;;("\\<[A-Za-z][A-Za-z_0-9]*:" . '-assigned-identifier-face)
        ("\\(\\<[A-Z][A-Za-z_0-9]*\\)\\(\\'\\|[^A-Za-z_0-9:]\\)" . (1 '-identifier-verb-face))
        ("->\\|;" . '-punctuation-face)
        (,-symbol-verbs . '-symbol-verb-face)
        (,-symbol-adverbs . '-adverb-face)
        ("\\<\\(p\\|q\\)\\>" . '-adverb-face)
        ("\\(\\`\\|[^A-Za-z_0-9]\\)\\(:\\)" . (2 '-punctuation-face))
        ("\"\\([^\"\\\\]\\|\\\\.\\)*\"" . 'font-lock-string-face)))

(define-derived-mode -mode fundamental-mode "-mode"
  "major mode for editing - code."
  (set-syntax-table -syntax-table)
  (setq font-lock-defaults '(-mode-fontlock)))
