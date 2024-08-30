use std::{
    collections::HashMap,
    fmt::{Display, Formatter, Error},
};

use crate::util::*;

pub struct Lexer {
    // Tokens that are never a prefix of another token, other than tokens also
    // in this list. Longer-length tokens appear first so e.g. "->" lexes as
    // "->" and not "-" then ">".
    literal_symbol_tokens: Vec<(String, Token)>,

    // Primitives that would otherwise lex as identifiers.
    literal_identifier_tokens: HashMap<String, Token>,
}

impl Lexer {
    pub fn new() -> Self {
        Self {
            literal_symbol_tokens: literal_symbol_tokens(),
            literal_identifier_tokens: literal_identifier_tokens(),
        }
    }

    pub fn tokenize_to_vec(&self, text: &str) -> Result<Vec<Token>, String> {
        let mut tokens = Vec::with_capacity(text.len() / 4);  // Guess
        self.tokenize(text, &mut tokens)?;
        Ok(tokens)
    }

    // TODO better error than String
    pub fn tokenize(&self, mut text: &str, tokens: &mut Vec<Token>) -> Result<(), String> {
        use Token::*;
        let full_text_len = text.len();
        'next_token: loop {
            let is_after_whitespace = {
                let len = text.len();
                text = text.trim_start_matches(|c: char| c.is_whitespace() && c != '\n');
                // The first character in the input is considered to be "after
                // whitespace". If it's '[', it should start an array literal,
                // not an arg list.
                len == full_text_len || len != text.len() || tokens.last().is_some_and(|t| matches!(t, Newline))
            };
            if text.is_empty() { break }

            // Comments
            if text.starts_with(&"\\ ") {
                match text.find('\n') {
                    None => break,
                    Some(i) => { text = &text[i..]; continue }
                }
            }

            // Literal tokens/keywords
            for (expected, token) in &self.literal_symbol_tokens {
                if let Some(rest) = text.strip_prefix(expected) {
                    text = rest;
                    tokens.push(match token {
                        Colon if is_after_whitespace => ColonAfterWhitespace,
                        LBracket {..} => LBracket { after_whitespace: is_after_whitespace },
                        DotDot {..} => DotDot { before_whitespace: text.starts_with(|c: char| c.is_whitespace()) },
                        _ => token.clone(),
                    });
                    continue 'next_token;
                }
            }

            if let Some((number_token, after_number)) = lex_number(text)? {
                tokens.push(number_token);
                text = after_number;
                continue 'next_token;
            }

            let is_identifier_char = |c: char| c.is_alphabetic() || c.is_digit(10) || c == '_';

            // Identifiers
            if let Some((name, after_name)) = prefix(text, is_identifier_char) {
                let first_char = name.chars().next().unwrap();
                let token = if let Some(keyword) = self.literal_identifier_tokens.get(name) {
                    keyword.clone()
                } else if first_char.is_ascii_uppercase() {
                    let mut name = name.to_string();
                    // SAFETY: We know that index 0 is ASCII.
                    unsafe { name.as_bytes_mut()[0] = first_char.to_ascii_lowercase() as u8; }
                    UpperName(name)
                } else if first_char.is_ascii_lowercase() {
                    LowerName(name.to_string())
                } else {
                    return err!("Unexpected: {name:?}; identifiers must begin with an ASCII alphabetic character.");
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
                                    None => return err!("Expected two hex digits after `\\x' escape in string literal"),
                                },
                                _ => c2,  // Just ignore the escape.
                            },
                        }
                    }
                    literal.push(c);
                }
                return err!("Unterminated string literal");
            }

            let invalid_sample = text.chars().take(10).collect::<String>();
            return err!("Invalid syntax: {}...", invalid_sample.trim());
        }
        Ok(())
    }
}



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
            None => return err!("Invalid float literal: expected digits after `{}'",
                                       &start[..(start.len() - text.len())]),
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
            let mut coefficient = parse::<f64>(integer_part)?;
            if negative { coefficient = -coefficient }
            Token::FloatLit(coefficient * 10f64.powi(exponent))
        }
        Some(decimal_part) => {
            let mut coefficient = parse::<f64>(integer_part)? + decimal_part;
            if negative { coefficient = -coefficient }
            Token::FloatLit(coefficient * 10f64.powi(exponent))
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
    Load,  // load
    IfUpper,  // If
    IfLower,  // if
    DotDot { before_whitespace: bool },  // ..

    LParen,  // (
    RParen,  // )
    LBracket { after_whitespace: bool },  // [
    RBracket,  // ]
    LBrace,  // {
    RBrace,  // }
    Colon,  // : not preceded by whitespace
    ColonAfterWhitespace, // : preceded by whitespace
    Semicolon, // ;
    Newline,  // \n
    C0Lower,  // c0
    C0Upper,  // C0

    PrimVerb(PrimVerb),
    PrimAdverb(PrimAdverb),

    IntLit(i64),
    FloatLit(f64),
    StrLit(String),
    UpperName(String),  // The stored string is in lowercase
    LowerName(String),
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
    HashColon,  // #:
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

    P,
    Q,

    // Hidden primitives below; these have no string representation and
    // shouldn't be in the token enum. TODO move these to compilation.
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimAdverb {
    AtColon,  // @:
    Dot,  // .
    SingleQuote,  // '
    Backtick,  // `
    BacktickColon,  // `:
    Tilde,  // ~
    Backslash, // \
    BackslashColon, // \:
    P,  // p
    Q,  // q

    Underscore,  // _

    // TODO converge/do-times/do-while
}

impl Display for Token {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        use Token::*;
        match self {
            RightArrow => f.write_str("->"),
            Load => f.write_str("load"),
            IfUpper => f.write_str("If"),
            IfLower => f.write_str("if"),
            DotDot { before_whitespace } => if *before_whitespace {
                f.write_str(".. ")
            } else {
                f.write_str("..")
            }
            LParen => f.write_str("("),
            RParen => f.write_str(")"),
            LBracket { after_whitespace } => if *after_whitespace {
                f.write_str(" [")
            } else {
                f.write_str("[")
            }
            RBracket => f.write_str("]"),
            LBrace => f.write_str("{"),
            RBrace => f.write_str("}"),
            Colon => f.write_str(":"),
            ColonAfterWhitespace => f.write_str(" : "),
            Semicolon => f.write_str(";"),
            Newline => f.write_str("\n"),
            C0Lower => f.write_str("c0"),
            C0Upper => f.write_str("C0"),
            UpperName(name) => {
                for c in char::from_u32(name.as_bytes()[0] as u32).unwrap().to_uppercase() {
                    std::fmt::Write::write_char(f, c)?;
                }
                f.write_str(&name[1..])
            }
            LowerName(name) => f.write_str(name),
            PrimVerb(prim) => Display::fmt(prim, f),
            PrimAdverb(prim) => Display::fmt(prim, f),
            IntLit(i) => Display::fmt(i, f),
            FloatLit(float) => Display::fmt(float, f),
            StrLit(lit) => std::fmt::Debug::fmt(lit, f),
        }
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
            HashColon => "#:",
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
            P => "P",
            Q => "Q",
        };

        f.write_str(s)
    }
}

impl Display for PrimAdverb {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        use PrimAdverb::*;
        let s: &str = match self {
            AtColon => "@:",
            SingleQuote => "'",
            Backtick => "`",
            BacktickColon => "`:",
            Dot => ".",
            Tilde => "~",
            Backslash => "\\",
            BackslashColon => "\\:",
            P => "p",
            Q => "q",
            Underscore => "_",
        };

        f.write_str(s)
    }
}

fn literal_symbol_tokens() -> Vec<(String, Token)> {
    use Token::*;
    use crate::lex::PrimVerb::*;
    use crate::lex::PrimAdverb::*;
    let mut ret: Vec<_> = [
        Colon,
        LBrace,
        LBracket { after_whitespace: false },
        LParen,
        RBrace,
        RBracket,
        RParen,
        RightArrow,
        Semicolon,
        DotDot { before_whitespace: false },
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
        PrimVerb(HashColon),
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

        PrimAdverb(AtColon),
        PrimAdverb(Backtick),
        PrimAdverb(BacktickColon),
        PrimAdverb(Backslash),
        PrimAdverb(BackslashColon),
        PrimAdverb(Dot),
        PrimAdverb(SingleQuote),
        PrimAdverb(Tilde),
    ].iter().map(|t| (t.to_string(), t.clone())).collect();

    // It would be cool to replace this with a compile-time assertion that the
    // array is sorted, but it doesn't matter enough.
    ret.sort_unstable_by_key(|(s, _)| -(s.len() as i32));
    ret
}

fn literal_identifier_tokens() -> HashMap<String, Token> {
    [
        Token::C0Lower,
        Token::C0Upper,
        Token::IfUpper,
        Token::IfLower,
        Token::Load,
        Token::PrimVerb(PrimVerb::P),
        Token::PrimVerb(PrimVerb::Q),
        Token::PrimAdverb(PrimAdverb::P),
        Token::PrimAdverb(PrimAdverb::Q),
        Token::PrimAdverb(PrimAdverb::Underscore),
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
