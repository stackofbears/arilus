use crate::lex::*;
use crate::lex::PrimVerb;

// TODO better error type than string
// Ok(Some(_)): Successfuly parsed an A
// Ok(None): Found no A and consumed no input
// Err(_): Parse failed
pub type Parsed<A> = Result<Option<A>, String>;
type Many<A> = Result<Vec<A>, String>;

pub fn parse(tokens: Vec<Token>) -> Many<Expr> {
    let mut parser = Parser::new(tokens);
    let parsed = parser.parse_exprs()?;
    if parser.token_index < parser.tokens.len() - 1 {
        return Err(format!("Unexpected {}; expected expressions", parser.tokens[0]));
    }
    Ok(parsed)
}

pub type Program = Vec<Expr>;

#[derive(Debug, Clone)]
pub enum Expr {
    Noun(Noun),
    Verb(Verb),
}

#[derive(Debug, Clone)]
pub enum Noun {
    LowerAssign(String, Box<Noun>),
    SmallNoun(SmallNoun),
    Sentence(SmallNoun, Vec<Predicate>),
}

#[derive(Debug, Clone)]
pub enum Predicate {
    VerbCall(Verb, Option<SmallNoun>),
    ForwardAssignment(String),
}

// TODO verb forward assignment?
#[derive(Debug, Clone)]
pub enum Verb {
    UpperAssign(String, Box<Verb>),
    SmallVerb(SmallVerb),
}

#[derive(Debug, Clone)]
pub enum SmallNoun {
    LowerName(String),
    Block(Vec<Expr>),  // parenthesized
    IntLiteral(i64),
    StringLiteral(String),
    ArrayLiteral(Vec<Expr>),
}

#[derive(Debug, Clone)]
pub enum SmallVerb {
    UpperName(String),
    PrimVerb(PrimVerb),
    // TODO explicit args
    Lambda(Vec<Expr>),
    Adverb(PrimAdverb, Box<SmallExpr>),
}

#[derive(Debug, Clone)]
pub enum SmallExpr {
    Noun(SmallNoun),
    Verb(SmallVerb),
}

struct Parser {
    tokens: Vec<Token>,
    token_index: usize,
}

use Noun::*;
use Verb::*;
use crate::parse::SmallNoun::*;
use crate::parse::SmallVerb::*;

impl Parser {
    fn new(tokens: Vec<Token>) -> Self {
        Parser { tokens, token_index: 0 }
    }

    fn peek(&self) -> Option<&Token> { self.tokens.get(self.token_index) }
    fn skip(&mut self) { self.token_index += 1; }

    // TODO lex newlines, right now semicolons are required
    fn parse_exprs(&mut self) -> Many<Expr> {
        let mut exprs = vec![];  // TODO reserve
        while let Some(expr) = self.parse_expr()? {
            exprs.push(expr);
            if let Some(Token::Semicolon) = self.peek() { self.skip(); }
        }
        Ok(exprs)
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
        let small_noun = match self.parse_small_noun()? {
            Some(small_noun) => small_noun,
            None => return Ok(None),
        };

        let noun = match self.peek() {
            Some(Token::Colon) => {
                self.skip();
                match small_noun {
                    LowerName(name) => match self.parse_noun()? {
                        Some(rhs) => LowerAssign(name, Box::new(rhs)),
                        None => return Err(format!("Unexpected end of input; expected RHS of assignment"))
                    }
                    _ => return Err(format!("Invalid noun assignment target: {small_noun:?}")),
                }
            }
            None => SmallNoun(small_noun),
            _ => Sentence(small_noun, self.parse_predicate()?),
        };

        Ok(Some(noun))
    }

    fn parse_small_noun(&mut self) -> Parsed<SmallNoun> {
        let small_noun = match self.peek() {
            Some(Token::LowerName(name)) => {
                let noun = LowerName(name.clone());
                self.skip();
                noun
            }
            Some(Token::IntLit(int)) => {
                let noun = IntLiteral(*int);
                self.skip();
                noun
            }
            Some(Token::StrLit(s)) => {
                let noun = StringLiteral(s.clone());
                self.skip();
                noun
            }
            Some(Token::LParen) => {
                self.skip();
                let exprs = self.parse_exprs()?;
                match self.peek() {
                    Some(Token::RParen) => {
                        self.skip();
                        Block(exprs)
                    }
                    Some(bad) => return Err(format!("Unexpected `{bad}'; expected `)'")),
                    None => return Err(format!("Unexpected end of input; expected `)'")),
                }
            }
            Some(Token::LBracket) => {
                self.skip();
                let elems = self.parse_exprs()?;
                match self.peek() {
                    Some(Token::RBracket) => {
                        self.skip();
                        ArrayLiteral(elems)
                    }
                    Some(bad) => return Err(format!("Unexpected `{bad}'; expected `]'")),
                    None => return Err(format!("Unexpected end of input; expected `]'")),
                }
            }                        
            _ => return Ok(None),
        };

        Ok(Some(small_noun))
    }

    fn parse_predicate(&mut self) -> Many<Predicate> {
        let mut predicates = vec![];
        loop {
            if let Some(Token::RightArrow) = self.peek() {
                self.skip();
                match self.peek() {
                    Some(Token::LowerName(name)) => {
                        predicates.push(Predicate::ForwardAssignment(name.clone()));
                        self.skip();
                    }
                    Some(bad) => return Err(format!("Unexpected `{bad}'; expected a name after `->'")),
                    None => return Err(format!("Unexpected end of input; expected a name after `->'")),
                }
            } else if let Some(verb) = self.parse_verb()? {
                predicates.push(Predicate::VerbCall(verb, self.parse_small_noun()?));
            } else {
                break;
            }
        }
        Ok(predicates)
    }

    fn parse_verb(&mut self) -> Parsed<Verb> {
        let small_verb = match self.parse_small_verb()? {
            Some(small_verb) => small_verb,
            None => return Ok(None),
        };

        let verb = match self.peek() {
            Some(Token::Colon) => {
                self.skip();
                match small_verb {
                    UpperName(name) => match self.parse_verb()? {
                        Some(rhs) => UpperAssign(name, Box::new(rhs)),
                        None => return Err(format!("Unexpected end of input; expected RHS of assignment")),
                    }
                    _ => return Err(format!("Invalid verb assignment target: {small_verb:?}")),
                }
            }
            _ => SmallVerb(small_verb),
        };

        Ok(Some(verb))
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
                // TODO explicit args
                let exprs = self.parse_exprs()?;
                match self.peek() {
                    Some(Token::RBrace) => {
                        let lambda = Lambda(exprs);
                        self.skip();
                        lambda
                    }
                    Some(bad) => return Err(format!("Unexpected `{bad}'; expected `}}'")),
                    None => return Err(format!("Unexpected end of input; expected `}}'")),
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
                    None => return Err(format!("Unexpected end of input; expected operand for adverb `{prim_adverb}'")),
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

fn require<A>(expected: &str, p: Parsed<A>) -> Result<A, String> {
    p.and_then(|opt| opt.ok_or_else(|| format!("Error: expected {}", expected)))
}
