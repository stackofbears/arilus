use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Error},
};

// TODO unused but possible tokens:
//   ## (length of length / length of take / train:take of length (possible) / train:take of take (possible but unlikely (pad w/0)))
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Token {
    // TODO floats, DoubleQuestion (??), BangEqual (!=)

    RightArrow,  // ->

    LParen,  // (
    RParen,  // )
    LBracket,  // [
    RBracket,  // ]
    LBrace,  // {
    RBrace,  // }
    Colon,  // :
    Semicolon, // ;
    Backtick,  // `

    PrimNoun(PrimNoun),
    PrimVerb(PrimVerb),
    PrimAdverb(PrimAdverb),

    IntLit(i64),
    StrLit(String),
    UpperName(String),  // The stored string is in lowercase
    LowerName(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PrimNoun {
    Print,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PrimVerb {
    DoublePipe, // ||
    DoubleAmpersand, // &&
    At,  // @
    Comma,  // ,
    Plus,   // +
    Minus,  // -
    Asterisk,  // *
    Hash,   // #
    Slash,  // /
    Caret,  // ^
    Pipe,   // |
    Bang,  // !
    Dollar,  // $
    Equals,  // =
    LessThan,  // <
    GreaterThan,  // >
    Percent,   // %
    Question,  // ?
    Ampersand, // &

    Print,

    // Hidden primitives below; these have no string representation and
    // shouldn't be in the token enum. TODO move these to compilation.
    Snoc,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum PrimAdverb {
    Dot,  // .
    SingleQuote,  // '
    Tilde,  // ~
    Backslash, // \
    // TODO converge/do-times/do-while
}

// TODO better error than String
// TODO comments
pub fn tokenize(mut text: &str) -> Result<Vec<Token>, String> {
    use Token::*;
    let literal_symbols = literal_symbol_tokens();
    let literal_identifiers = literal_identifier_tokens();
    let mut tokens = Vec::with_capacity(text.len() / 4);  // Guess
    'next_token: loop {
        text = text.trim_start_matches(|c: char| c.is_whitespace());
        if text.is_empty() { break }

        // Literal tokens/keywords
        for (expected, token) in &literal_symbols {
            if let Some(rest) = text.strip_prefix(expected) {
                text = rest;
                tokens.push(token.clone());
                continue 'next_token;
            }
        }

        // Numbers
        // TODO exponential notation
        if let Some((number, after_number)) = prefix(text, |c: char| c.is_digit(10)) {
            let int = number.parse::<i64>().map_err(|err| err.to_string())?;
            tokens.push(IntLit(int));
            text = after_number;
            continue 'next_token;
        }

        // Identifiers
        if let Some((name, after_name)) = prefix(text, |c: char| c.is_alphabetic() || c.is_digit(10) || c == '_') {
            let first_char = name.chars().next().unwrap();
            let token = if let Some(keyword) = literal_identifiers.get(name) {
                keyword.clone()
            } else if first_char.is_ascii_uppercase() {
                let mut name = name.to_string();
                // SAFETY: We know that index 0 is ASCII.
                unsafe { name.as_bytes_mut()[0] = first_char.to_ascii_lowercase() as u8; }
                UpperName(name)
            } else if first_char.is_ascii_lowercase() {
                LowerName(name.to_string())
            } else {
                return Err(format!("Unexpected: {name:?}; identifiers must begin with an ASCII alphabetic character."));
            };
            tokens.push(token);
            text = after_name;
            continue 'next_token;
        }
        
        // String literals
        if let Some(after_quote) = text.strip_prefix("\"") {
            let mut chars = after_quote.chars();
            let mut literal = String::new();
            while let Some(mut c) = chars.next() {
                if c == '"' {
                    tokens.push(StrLit(literal));
                    text = chars.as_str();
                    continue 'next_token;
                }
                if c == '\\' {
                    if let Some(c2) = chars.next() { c = c2 } else { break }
                }
                literal.push(c);
            }
            return Err(format!("Unterminated string literal"));
        }

        return Err(format!("Invalid syntax: {}...", &text[0..10.min(text.len())]));
    }

    Ok(tokens)
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        use Token::*;
        match self {
            RightArrow => f.write_str("->"),
            LParen => f.write_str("("),
            RParen => f.write_str(")"),
            LBracket => f.write_str("["),
            RBracket => f.write_str("]"),
            LBrace => f.write_str("{"),
            RBrace => f.write_str("}"),
            Colon => f.write_str(":"),
            Semicolon => f.write_str(";"),
            Backtick => f.write_str("`"),
            UpperName(name) => f.write_str(name),
            LowerName(name) => f.write_str(name),
            PrimNoun(prim) => Display::fmt(prim, f),
            PrimVerb(prim) => Display::fmt(prim, f),
            PrimAdverb(prim) => Display::fmt(prim, f),
            IntLit(i) => Display::fmt(i, f),
            StrLit(lit) => std::fmt::Debug::fmt(lit, f),
        }
    }
}

impl Display for PrimNoun {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        use PrimNoun::*;
        let s: &str = match self {
            Print => "print",
        };

        f.write_str(s)
    }
}

impl Display for PrimVerb {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        use PrimVerb::*;
        let s: &str = match self {
            DoublePipe => "||",
            DoubleAmpersand => "&&",
            At => "@",
            Comma => ",",
            Plus => "+",
            Minus => "-",
            Asterisk => "*",
            Hash => "#",
            Slash => "/",
            Caret => "^",
            Pipe => "|",
            Bang => "!",
            Dollar => "$",
            Equals => "=",
            LessThan => "<",
            GreaterThan => ">",
            Percent => "%",
            Question => "?",
            Ampersand => "&",
            Print => "Print",

            // The verbs below are technically hidden from the user.
            Snoc => "Snoc",
        };

        f.write_str(s)
    }
}

impl Display for PrimAdverb {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        use PrimAdverb::*;
        let s: &str = match self {
            SingleQuote => "'",
            Dot => ".",
            Tilde => "~",
            Backslash => "\\",
        };

        f.write_str(s)
    }
}

// These tokens are never a prefix of another token, except for tokens that are
// also in this list. Longer-length tokens appear first so e.g. "->" lexes as
// "->" and not "-" then ">".
fn literal_symbol_tokens() -> Vec<(String, Token)> {
    use Token::*;
    use crate::lex::PrimVerb::*;
    use crate::lex::PrimAdverb::*;
    let mut ret: Vec<_> = [
        Backtick,
        Colon,
        LBrace,
        LBracket,
        LParen,
        RBrace,
        RBracket,
        RParen,
        RightArrow,
        Semicolon,

        PrimVerb(Ampersand),
        PrimVerb(Asterisk),
        PrimVerb(At),
        PrimVerb(Bang),
        PrimVerb(Caret),
        PrimVerb(Comma),
        PrimVerb(Dollar),
        PrimVerb(DoubleAmpersand),
        PrimVerb(DoublePipe),
        PrimVerb(Equals),
        PrimVerb(GreaterThan),
        PrimVerb(Hash),
        PrimVerb(LessThan),
        PrimVerb(Minus),
        PrimVerb(Percent),
        PrimVerb(Pipe),
        PrimVerb(Plus),
        PrimVerb(Question),
        PrimVerb(Slash),

        PrimAdverb(Backslash),
        PrimAdverb(Dot),
        PrimAdverb(SingleQuote),
        PrimAdverb(Tilde),
    ].iter().map(|t| (t.to_string(), t.clone())).collect();

    // It would be cool to replace this with a compile-time assertion that the
    // array is sorted, but it doesn't matter enough.
    ret.sort_unstable_by_key(|(s, _)| -(s.len() as i32));
    ret
}

// These are primitives that would otherwise lex as identifiers.
fn literal_identifier_tokens() -> HashMap<String, Token> {
    [
        Token::PrimNoun(PrimNoun::Print),
        Token::PrimVerb(PrimVerb::Print),
    ].iter().map(|t| (t.to_string(), t.clone())).collect()
}

fn prefix<F: FnMut(char) -> bool>(s: &str, mut pred: F) -> Option<(&str, &str)> {
    match s.find(|c| !pred(c)) {
        Some(0) => None,
        Some(i) => Some(s.split_at(i)),
        None => Some((s, "")),
    }
}
