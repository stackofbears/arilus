;;; Code:

(defgroup arilus-faces nil "Faces for arilus-mode")

(defface arilus-keyword-face
    `((t (:foreground "DodgerBlue2")))
  "Face used to highlight keywords, like `load'"
  :group 'arilus-faces)

(defface arilus-assigned-identifier-face
    `((t (:foreground "medium purple")))
  "Face used to highlight identifiers being assigned to with a colon"
  :group 'arilus-faces)

(defface arilus-symbol-verb-face
    `((t (:foreground "HotPink2")))
  "Face used to highlight verbs with symbolic names"
  :group 'arilus-faces)

(defface arilus-identifier-verb-face
    `((t (:foreground "plum3")))
  "Face used to highlight verbs with identifier names"
  :group 'arilus-faces)

(defface arilus-adverb-face
    `((t (:foreground "medium orchid")))
  "Face used to highlight primitive adverbs"
  :group 'arilus-faces)

(defface arilus-punctuation-face
    `((t (:foreground "cornflower blue")))
  "Face used to highlight punctuation, like ; and ->"
  :group 'arilus-faces)

(defconst arilus-keywords
  (regexp-opt (split-string "load if If rec Rec") 'words)
  "Keywords")

(defconst arilus-syntax-table
  (let ((table (make-syntax-table)))
    ;; word
    (modify-syntax-entry ?_ "w" table)
    ;; punctuation
    (modify-syntax-entry ?$ "." table)
    (modify-syntax-entry ?% "." table)
    table))

(defconst arilus-symbol-verbs-len-1
  (regexp-opt (split-string "- @ , + * # / ^ | ! $ = < > % ? &")))

(defconst arilus-symbol-verbs-len-2
  (regexp-opt (split-string "|| && == =! -: <: >: #: // <= >= ?: ,: .:")))

(defconst arilus-symbol-adverbs-len-1
  (regexp-opt (split-string ". ' ` ~ \\")))

(defconst arilus-symbol-adverbs-len-2
  (regexp-opt (split-string "\\: `: @:")))

;; TODO It would be nice to highlight colon assignments in a different color,
;; but we can't parse the LHS with a regex because it's an arbitrarily-nested
;; pattern.
(setq arilus-mode-fontlock
      `((,arilus-keywords . 'arilus-punctuation-face)

        ;; Need OVERRIDE=t or else comments containing string literals won't
        ;; highlight properly.
        ("\\\\ .*" . (0 'font-lock-comment-face t))

        ("\\(\\<[A-Z][A-Za-z_0arilus-9]*\\)\\(\\'\\|[^A-Za-z_0-9:]\\)" . (1 'arilus-identifier-verb-face))
        ("->\\|;\\|\\[\\|\\]" . 'arilus-punctuation-face)
        (,arilus-symbol-verbs-len-2 . 'arilus-symbol-verb-face)
        (,arilus-symbol-adverbs-len-2 . 'arilus-adverb-face)
        (,arilus-symbol-verbs-len-1 . 'arilus-symbol-verb-face)
        (,arilus-symbol-adverbs-len-1 . 'arilus-adverb-face)
        ("\\<\\(p\\|q\\|_\\)\\>" . 'arilus-adverb-face)
        ("\\(\\`\\|[^A-Za-z_0-9]\\)\\(:\\)" . (2 'arilus-punctuation-face))
        ("\"\\([^\"\\\\]\\|\\\\.\\)*\"" . 'font-lock-string-face)))

(define-derived-mode arilus-mode fundamental-mode "arilus-mode"
  "Major mode for editing Arilus code."
  (set-syntax-table arilus-syntax-table)
  (setq font-lock-defaults '(arilus-mode-fontlock))
  (setq comment-start "\\ "))
