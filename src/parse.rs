use crate::lex;
use crate::lex::*;

// TODO better error type than string

// Ok(Some(_)): Successfuly parsed an A
// Ok(None): Found no A and consumed no input
// Err(_): Parse failed
pub type Parsed<A> = Result<Option<A>, String>;
pub type Many<A> = Result<Vec<A>, String>;

pub fn parse(tokens: &[Token]) -> Many<Expr> {
    let mut parser = Parser::new(tokens);
    let parsed = parser.parse_exprs()?;
    if parser.token_index < parser.tokens.len() - 1 {
        return Err(format!("Unexpected token `{}'; expected `;', newline, or end of input",
                           parser.tokens[parser.token_index]));
    }
    Ok(parsed)
}

#[derive(Debug, Clone)]
pub enum Expr {
    Noun(Noun),
    Verb(Verb),
}

#[derive(Debug, Clone)]
pub enum Noun {
    LowerAssign(Pattern, Box<Noun>),
    SmallNoun(SmallNoun),
    Sentence(SmallNoun, Vec<Predicate>),
}

#[derive(Debug, Clone)]
pub enum Predicate {
    VerbCall(Verb, Option<SmallNoun>),
    ForwardAssignment(Pattern),
}

// TODO verb forward assignment?
#[derive(Debug, Clone)]
pub enum Verb {
    UpperAssign(String, Box<Verb>),
    SmallVerb(SmallVerb),
}

#[derive(Debug, Clone)]
pub enum SmallNoun {
    PrimNoun(lex::PrimNoun),
    LowerName(String),
    Block(Vec<Expr>),  // parenthesized
    IntLiteral(i64),
    FloatLiteral(f64),
    CharLiteral(u8),
    StringLiteral(String),
    ArrayLiteral(Vec<Expr>),
}

#[derive(Debug, Clone)]
pub enum SmallVerb {
    UpperName(String),
    PrimVerb(lex::PrimVerb),
    Lambda(Option<ExplicitArgs>, Vec<Expr>),
    Adverb(PrimAdverb, Box<SmallExpr>),
}

#[derive(Debug, Clone)]
pub struct ExplicitArgs {
    pub x: Pattern,
    pub y: Option<Pattern>,
}

#[derive(Debug, Clone)]
pub enum Pattern {
    Name(String),
    Array(Vec<Pattern>),
    As(Box<Pattern>, Box<Pattern>),
}

#[derive(Debug, Clone)]
pub enum SmallExpr {
    Noun(SmallNoun),
    Verb(SmallVerb),
}

struct Parser<'a> {
    tokens: &'a [Token],
    token_index: usize,
}

use Noun::*;
use Verb::*;
use crate::parse::SmallNoun::*;
use crate::parse::SmallVerb::*;

impl<'a> Parser<'a> {
    fn new(tokens: &'a [Token]) -> Self {
        Parser { tokens, token_index: 0 }
    }

    fn peek(&self) -> Option<&Token> { self.tokens.get(self.token_index) }
    fn skip(&mut self) { self.token_index += 1; }

    fn expected(&self, expected: &str) -> String {
        match self.peek() {
            Some(Token::Newline) => format!("Unexpected newline; expected {expected}"),
            Some(bad) => format!("Unexpected `{bad}'; expected {expected}"),
            None => format!("Unexpected end of input; expected {expected}"),
        }
    }

    // True if any newlines were skipped.
    fn skip_newlines(&mut self) -> bool {
        let before = self.token_index;
        while let Some(Token::Newline) = self.peek() { self.skip() }
        self.token_index > before
    }

    fn parse_exprs(&mut self) -> Many<Expr> {
        self.parse_sequenced(Self::parse_expr)
    }

    fn parse_pattern_list(&mut self) -> Many<Pattern> {
        self.parse_sequenced(Self::parse_pattern)
    }

    fn consume(&mut self, tok: Token) -> bool {
        if self.peek() == Some(&tok) { self.skip(); true }
        else { false }
    }

    fn parse_sequenced<A, F: Fn(&mut Self) -> Parsed<A>>(&mut self, parse: F) -> Many<A> {
        self.skip_newlines();
        let mut ret = vec![];  // TODO reserve
        let mut next_required = false;
        loop {
            match parse(self)? {
                Some(parsed) => ret.push(parsed),
                None if next_required => return Err(self.expected(&"expression")),
                _ => break,
            }

            next_required = self.consume(Token::Semicolon);
            let next_allowed = next_required | self.skip_newlines();
            if !next_allowed { break }
        }
        Ok(ret)
    }

    fn parse_expr(&mut self) -> Parsed<Expr> {
        let expr = if let Some(noun) = self.parse_noun()? {
            Expr::Noun(noun)
        } else if let Some(verb) = self.parse_verb()? {
            Expr::Verb(verb)
        } else {
            return Ok(None)
        };
        Ok(Some(expr))
    }

    fn parse_noun(&mut self) -> Parsed<Noun> {
        let before_small_noun = self.token_index;
        let small_noun = match self.parse_small_noun()? {
            Some(small_noun) => small_noun,
            None => return Ok(None),
        };

        let noun = match self.peek() {
            Some(Token::Colon) => {
                self.token_index = before_small_noun;
                let pattern = match self.parse_pattern()? {
                    Some(pattern) => pattern,
                    None => return Err(self.expected(&"pattern")),
                };
                assert!(self.consume(Token::Colon));
                match self.parse_noun()? {
                    Some(rhs) => LowerAssign(pattern, Box::new(rhs)),
                    None => return Err(self.expected(&"RHS of noun assignment")),
                }
            }
            None => SmallNoun(small_noun),
            _ => Sentence(small_noun, self.parse_predicate()?),
        };

        Ok(Some(noun))
    }

    fn parse_small_noun(&mut self) -> Parsed<SmallNoun> {
        let small_noun = match self.parse_small_noun_no_stranding()? {
            None => return Ok(None),
            Some(small_noun) => small_noun,
        };
        match self.parse_small_noun_no_stranding()? {
            None => Ok(Some(small_noun)),
            Some(next_small_noun) => {
                let mut stranded_nouns = vec![Expr::Noun(Noun::SmallNoun(small_noun)),
                                              Expr::Noun(Noun::SmallNoun(next_small_noun))];
                while let Some(next_small_noun) = self.parse_small_noun_no_stranding()? {
                    stranded_nouns.push(Expr::Noun(Noun::SmallNoun(next_small_noun)));
                }
                Ok(Some(ArrayLiteral(stranded_nouns)))
            }
        }
    }

    fn parse_small_noun_no_stranding(&mut self) -> Parsed<SmallNoun> {
        // TODO prim nouns
        let small_noun = match self.peek() {
            Some(&Token::PrimNoun(prim)) => {
                self.skip();
                PrimNoun(prim)
            }
            Some(Token::LowerName(name)) => {
                let name_clone = name.clone();
                self.skip();
                LowerName(name_clone)
            }
            Some(&Token::IntLit(int)) => {
                self.skip();
                IntLiteral(int)
            }
            Some(&Token::FloatLit(float)) => {
                self.skip();
                FloatLiteral(float)
            }
            Some(Token::StrLit(s)) => {
                let noun = if s.len() == 1 {
                    CharLiteral(s.as_bytes()[0])  // TODO unicode
                } else {
                    StringLiteral(s.clone())
                };
                self.skip();
                noun
            }
            Some(Token::LParen) => {
                self.skip();
                let exprs = self.parse_exprs()?;
                if self.consume(Token::RParen) {
                    Block(exprs)
                } else {
                    return Err(self.expected(&"`)'"))
                }
            }
            Some(Token::LBracket) => {
                self.skip();
                let elems = self.parse_exprs()?;
                if self.consume(Token::RBracket) {
                    ArrayLiteral(elems)
                } else {
                    return Err(self.expected(&"`]'"))
                }
            }
            _ => return Ok(None),
        };
        Ok(Some(small_noun))
    }

    fn parse_predicate(&mut self) -> Many<Predicate> {
        let mut predicates = vec![];
        loop {
            if self.consume(Token::RightArrow) {
                match self.parse_pattern()? {
                    Some(pat) => predicates.push(Predicate::ForwardAssignment(pat)),
                    None => return Err(self.expected(&"pattern after `->'")),
                }
            } else if let Some(verb) = self.parse_verb()? {
                predicates.push(Predicate::VerbCall(verb, self.parse_small_noun()?))
            } else {
                break
            }
        }
        Ok(predicates)
    }

    fn parse_verb(&mut self) -> Parsed<Verb> {
        let small_verb = match self.parse_small_verb()? {
            Some(small_verb) => small_verb,
            None => return Ok(None),
        };

        let verb = if self.consume(Token::Colon) {
            match small_verb {
                UpperName(name) => match self.parse_verb()? {
                    Some(rhs) => UpperAssign(name, Box::new(rhs)),
                    None => return Err(self.expected(&"RHS of verb assignment")),
                }
                _ => return Err(format!("Invalid verb assignment target: {small_verb:?}")),
            }
        } else {
            SmallVerb(small_verb)
        };

        Ok(Some(verb))
    }

    fn parse_small_pattern(&mut self) -> Parsed<Pattern> {
        // TODO UpperName?
        match self.peek() {
            Some(Token::LowerName(name)) => {
                let pat = Pattern::Name(name.clone());
                self.skip();
                Ok(Some(pat))
            }
            Some(Token::LBracket) => {
                self.skip();
                let names = self.parse_pattern_list()?;
                if self.consume(Token::RBracket) {
                    Ok(Some(Pattern::Array(names)))
                } else {
                    return Err(self.expected(&"`]' to end pattern"))
                }
            }
            Some(Token::LParen) => {
                self.skip();
                let inner = match self.parse_pattern()? {
                    Some(pat) => pat,
                    None => return Err(self.expected(&"pattern")),
                };

                if self.consume(Token::RParen) {
                    Ok(Some(inner))
                } else {
                    return Err(self.expected(&"`)'"))
                }
            }
            _ => return Ok(None),
        }
    }

    fn parse_pattern(&mut self) -> Parsed<Pattern> {
        let mut pat = match self.parse_small_pattern()? {
            None => return Ok(None),
            Some(small_pat) => small_pat,
        };

        if let Some(next_small_pat) = self.parse_small_pattern()? {
            let mut stranded_pats = vec![pat, next_small_pat];
            while let Some(next_small_pat) = self.parse_small_pattern()? {
                stranded_pats.push(next_small_pat)
            }
            pat = Pattern::Array(stranded_pats)
        }

        while self.consume(Token::RightArrow) {
            match self.parse_pattern()? {
                Some(next_pat) => pat = Pattern::As(Box::new(pat), Box::new(next_pat)),
                None => return Err(self.expected(&"pattern after `->'")),
            }
        }

        Ok(Some(pat))
    }

    fn parse_explicit_args(&mut self) -> Parsed<ExplicitArgs> {
        self.skip_newlines();
        if !self.consume(Token::PrimVerb(lex::PrimVerb::Pipe)) {
            return Ok(None)
        }

        self.skip_newlines();
        let x = match self.parse_pattern()? {
            Some(pat) => pat,
            None => return Err(self.expected(&"pattern for explicit left argument")),
        };

        let y = {
            let y_required = self.consume(Token::Semicolon);
            let y_allowed = y_required | self.skip_newlines();

            if y_allowed {
                match self.parse_pattern()? {
                    Some(pat) => Some(pat),
                    None if y_required => return Err(self.expected(&"pattern for explicit right argument")),
                    None => None,
                }
            } else {
                None
            }
        };

        self.skip_newlines();
        if !self.consume(Token::PrimVerb(lex::PrimVerb::Pipe)) {
            return Err(self.expected(&"`|' to end explicit argument list"))
        }

        Ok(Some(ExplicitArgs { x, y }))
    }

    fn parse_small_verb(&mut self) -> Parsed<SmallVerb> {
        let small_verb = match self.peek() {
            Some(Token::UpperName(name)) => {
                let upper_name = UpperName(name.clone());
                self.skip();
                upper_name
            }
            Some(Token::LBrace) => {
                self.skip();
                let explicit_args = self.parse_explicit_args()?;
                let exprs = self.parse_exprs()?;
                if exprs.is_empty() {  // TODO remove to enable () parse
                    return Err(self.expected(&"expression"))
                }
                if self.consume(Token::RBrace) {
                    Lambda(explicit_args, exprs)
                } else {
                    return Err(self.expected("`;', newline, or `}}'"))
                }
            }
            Some(Token::PrimVerb(prim_verb)) => {
                let prim = PrimVerb(*prim_verb);
                self.skip();
                prim
            }
            Some(&Token::PrimAdverb(prim_adverb)) => {
                self.skip();
                match self.parse_small_expr()? {
                    Some(adverb_operand) => Adverb(prim_adverb, Box::new(adverb_operand)),
                    None => return Err(self.expected(&format!("operand for adverb `{prim_adverb}'"))),
                }
            }
            _ => return Ok(None),
        };

        Ok(Some(small_verb))
    }

    fn parse_small_expr(&mut self) -> Parsed<SmallExpr> {
        let small_expr = if let Some(small_noun) = self.parse_small_noun()? {
            SmallExpr::Noun(small_noun)
        } else if let Some(small_verb) = self.parse_small_verb()? {
            SmallExpr::Verb(small_verb)
        } else {
            return Ok(None)
        };
        Ok(Some(small_expr))
    }
}
