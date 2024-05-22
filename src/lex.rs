use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Error},
};

fn lex_number(mut text: &str) -> Result<Option<(Token, &str)>, String> {
    use std::str::FromStr;
    // TODO hex/arbitrary base
    fn parse<A: FromStr>(s: &str) -> Result<A, String>
    where A::Err: Display {
        s.parse::<A>().map_err(|err| err.to_string())
    }

    let start = text;

    let negative = text.starts_with("_");
    if negative { text = &text[1..] }

    let integer_part = match prefix(text, |c: char| c.is_digit(10)) {
        Some((integer_part, after_digits)) => {
            text = after_digits;
            integer_part
        }
        _ => return Ok(None),
    };

    let decimal_part = if text.starts_with('.') {
        text = &text[1..];
        match prefix(&text, |c: char| c.is_digit(10)) {
            None => Some(0.),
            Some((decimal, rest)) => {
                text = rest;
                Some(
                    parse::<f64>(decimal)? * 10f64.powi(-(decimal.len() as i32))
                )
            }
        }
    } else {
        None
    };

    let exponent = if text.starts_with('e') {
        text = &text[1..];
        let negative_exponent = text.starts_with("_");
        if negative_exponent { text = &text[1..] }
        match prefix(&text, |c: char| c.is_digit(10)) {
            Some((suffix, rest)) => {
                text = rest;
                let exponent = parse::<i32>(suffix)?;
                if negative_exponent { -exponent } else { exponent }
            }
            None => return Err(format!("Invalid float literal: expected digits after `{}'",
                                       &start[..(start.len() - text.len())])),
        }
    } else {
        0
    };

    let token = match decimal_part {
        // Integer
        None if exponent >= 0 => {
            let mut int = parse::<i64>(integer_part)?;
            if negative { int = -int }
            Token::IntLit(int * 10i64.pow(exponent as u32))
        }
        // Float
        None => {
            let mut int = parse::<f64>(integer_part)?;
            if negative { int = -int }
            Token::FloatLit(int * 10f64.powi(exponent))
        }
        Some(decimal_part) => {
            let mut int = parse::<f64>(integer_part)?;
            if negative { int = -int }
            Token::FloatLit((int + decimal_part) * 10f64.powi(exponent))
        }
    };

    Ok(Some((token, text)))
}

// TODO unused but possible tokens:
//   ## (length of length / length of take / train:take of length (possible) / train:take of take (possible but unlikely (pad w/0)))
#[derive(Debug, Clone, PartialEq)]
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
    Newline,  // \n

    PrimNoun(PrimNoun),
    PrimVerb(PrimVerb),
    PrimAdverb(PrimAdverb),

    IntLit(i64),
    FloatLit(f64),
    StrLit(String),
    UpperName(String),  // The stored string is in lowercase
    LowerName(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimNoun {
    Exit,
    Print,
    ReadFile,
    Rand,
    Rec,
    C0,  // The null character
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
// TODO -: remove
pub enum PrimVerb {
    DoublePipe,  // ||
    DoubleAmpersand,  // &&
    DoubleEquals,  // ==
    EqualBang,  // =!
    At,  // @
    Comma,  // ,
    Plus,   // +
    Minus,  // -
    Asterisk,  // *
    Hash,   // #
    Slash,  // /
    DoubleSlash,  // //
    Caret,  // ^
    Pipe,   // |
    Bang,  // !
    Dollar,  // $
    Equals,  // =
    LessThan,  // <
    LessThanEquals,  // <=
    LessThanColon,  // <:
    GreaterThan,  // >
    GreaterThanColon,  // >:
    GreaterThanEquals,  // >=
    Percent,   // %
    Question,  // ?
    QuestionColon,  // ?:
    Ampersand, // &

    Exit,
    Print,
    ReadFile,
    Rand,
    Rec,
    C0,

    // Hidden primitives below; these have no string representation and
    // shouldn't be in the token enum. TODO move these to compilation.

    // Currently used only for repl display, but this could be exposed once
    // prints in valid syntax (instead of Rust Debug).
    DebugPrint,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimAdverb {
    Dot,  // .
    SingleQuote,  // '
    Backtick,  // `
    Tilde,  // ~
    Backslash, // \
    // TODO converge/do-times/do-while
}

pub fn tokenize_to_vec(text: &str) -> Result<Vec<Token>, String> {
    let mut tokens = Vec::with_capacity(text.len() / 4);  // Guess
    tokenize(text, &mut tokens)?;
    Ok(tokens)
}

// TODO better error than String
// TODO comments
pub fn tokenize(mut text: &str, tokens: &mut Vec<Token>) -> Result<(), String> {
    use Token::*;
    let literal_symbols = literal_symbol_tokens();
    let literal_identifiers = literal_identifier_tokens();
    'next_token: loop {
        text = text.trim_start_matches(|c: char| c.is_whitespace() && c != '\n');
        if text.is_empty() { break }

        // Comments
        if text.starts_with(&"\\ ") {
            match text.find('\n') {
                None => break,
                Some(i) => { text = &text[i..]; continue }
            }
        }

        // Literal tokens/keywords
        for (expected, token) in &literal_symbols {
            if let Some(rest) = text.strip_prefix(expected) {
                text = rest;
                tokens.push(token.clone());
                continue 'next_token;
            }
        }

        if let Some((number_token, after_number)) = lex_number(text)? {
            tokens.push(number_token);
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
                    match chars.next() {
                        None => break,
                        // TODO more escapes, numeric escapes
                        Some(c2) => c = match c2 {
                            '"' => '"',
                            '\\' => '\\',
                            'n' => '\n',
                            't' => '\t',
                            'r' => '\r',
                            '0' => '\0',
                            'x' => match chars.next().and_then(hex_digit_to_value).zip(chars.next().and_then(hex_digit_to_value)) {
                                Some((hex1, hex2)) => char::from(hex1 * 16 + hex2),
                                None => return Err(format!("Expected two hex digits after `\\x' escape in string literal")),
                            },
                            _ => c2,  // Just ignore the escape.
                        },
                    }
                }
                literal.push(c);
            }
            return Err(format!("Unterminated string literal"));
        }

        return Err(format!("Invalid syntax: {}...", &text[0..10.min(text.len())]));
    }
    Ok(())
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
            Newline => f.write_str("\n"),
            UpperName(name) => {
                for c in char::from_u32(name.as_bytes()[0] as u32).unwrap().to_uppercase() {
                    std::fmt::Write::write_char(f, c)?;
                }
                f.write_str(&name[1..])
            }
            LowerName(name) => f.write_str(name),
            PrimNoun(prim) => Display::fmt(prim, f),
            PrimVerb(prim) => Display::fmt(prim, f),
            PrimAdverb(prim) => Display::fmt(prim, f),
            IntLit(i) => Display::fmt(i, f),
            FloatLit(float) => Display::fmt(float, f),
            StrLit(lit) => std::fmt::Debug::fmt(lit, f),
        }
    }
}

impl Display for PrimNoun {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        let s: &str = match self {
            PrimNoun::ReadFile => "readFile",
            PrimNoun::Exit => "exit",
            PrimNoun::Print => "print",
            PrimNoun::Rand => "rand",
            PrimNoun::Rec => "rec",
            PrimNoun::C0 => "c0",
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
            DoubleEquals => "==",
            EqualBang => "=!",
            LessThanColon => "<:",
            GreaterThanColon => ">:",
            At => "@",
            Comma => ",",
            Plus => "+",
            Minus => "-",
            Asterisk => "*",
            Hash => "#",
            Slash => "/",
            DoubleSlash => "//",
            Caret => "^",
            Pipe => "|",
            Bang => "!",
            Dollar => "$",
            Equals => "=",
            LessThan => "<",
            LessThanEquals => "<=",
            GreaterThan => ">",
            GreaterThanEquals => ">=",
            Percent => "%",
            Question => "?",
            QuestionColon => "?:",
            Ampersand => "&",
            Exit => "Exit",
            Print => "Print",
            ReadFile => "ReadFile",
            DebugPrint => "DebugPrint",
            Rand => "Rand",
            Rec => "Rec",
            C0 => "C0",
        };

        f.write_str(s)
    }
}

impl Display for PrimAdverb {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        use PrimAdverb::*;
        let s: &str = match self {
            SingleQuote => "'",
            Backtick => "`",
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
        Colon,
        LBrace,
        LBracket,
        LParen,
        RBrace,
        RBracket,
        RParen,
        RightArrow,
        Semicolon,
        Newline,

        PrimVerb(Ampersand),
        PrimVerb(Asterisk),
        PrimVerb(At),
        PrimVerb(Bang),
        PrimVerb(Caret),
        PrimVerb(Comma),
        PrimVerb(Dollar),
        PrimVerb(DoubleAmpersand),
        PrimVerb(DoublePipe),
        PrimVerb(DoubleEquals),
        PrimVerb(EqualBang),
        PrimVerb(Equals),
        PrimVerb(Hash),
        PrimVerb(LessThan),
        PrimVerb(LessThanEquals),
        PrimVerb(LessThanColon),
        PrimVerb(GreaterThan),
        PrimVerb(GreaterThanColon),
        PrimVerb(GreaterThanEquals),
        PrimVerb(Minus),
        PrimVerb(Percent),
        PrimVerb(Pipe),
        PrimVerb(Plus),
        PrimVerb(Question),
        PrimVerb(QuestionColon),
        PrimVerb(Slash),
        PrimVerb(DoubleSlash),

        PrimAdverb(Backtick),
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
        Token::PrimNoun(PrimNoun::Exit),
        Token::PrimVerb(PrimVerb::Exit),

        Token::PrimNoun(PrimNoun::Print),
        Token::PrimVerb(PrimVerb::Print),

        Token::PrimNoun(PrimNoun::ReadFile),
        Token::PrimVerb(PrimVerb::ReadFile),

        Token::PrimNoun(PrimNoun::Rand),
        Token::PrimVerb(PrimVerb::Rand),

        Token::PrimNoun(PrimNoun::Rec),
        Token::PrimVerb(PrimVerb::Rec),

        Token::PrimNoun(PrimNoun::C0),
        Token::PrimVerb(PrimVerb::C0),
    ].iter().map(|t| (t.to_string(), t.clone())).collect()
}

fn prefix<F: FnMut(char) -> bool>(s: &str, mut pred: F) -> Option<(&str, &str)> {
    match s.find(|c| !pred(c)) {
        Some(0) => None,
        Some(i) => Some(s.split_at(i)),
        None => Some((s, "")),
    }
}

fn hex_digit_to_value(c: char) -> Option<u8> {
    if c.is_ascii_digit() {
        Some(c as u8 - '0' as u8)
    } else if c.is_ascii_hexdigit() {
        Some(c.to_ascii_uppercase() as u8 - 'A' as u8 + 10)
    } else {
        None
    }
}
