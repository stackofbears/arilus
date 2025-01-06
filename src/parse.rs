use crate::bytecode::PrimFunc;
use crate::lex::*;
use crate::util::*;

// Ok(Some(_)): Successfuly parsed an A
// Ok(None): Found no A and consumed no input
// Err(_): Parse failed (consumed input but couldn't fully parse)
pub type Parsed<A> = Res<Option<A>>;

pub type Many<A> = Res<Vec<A>>;

pub fn parse(tokens: &[Token]) -> Many<Expr> {
    if tokens.is_empty() { return Ok(vec![]) }
    let mut parser = Parser::new(tokens);
    let parsed = parser.parse_exprs()?;
    if parser.token_index < parser.tokens.len() - 1 {
        return cold_err!("Unexpected `{}'; expected `;', newline, or end of input",
                         parser.tokens[parser.token_index]);
    }
    Ok(parsed)
}

#[derive(Debug, Clone)]
pub enum Expr {
    Noun(Noun),
    Verb(Verb),
    PragmaLoad(String),
}

#[derive(Debug, Clone)]
pub enum Noun {
    SmallNoun(SmallNoun),
    LowerAssign(Pattern, Box<Noun>),
    ModifyingAssign(Pattern, Vec<Predicate>),
    Sentence(SmallNoun, Vec<Predicate>),
}

#[derive(Debug, Clone)]
pub enum SmallNoun {
    If3(Box<Expr>, Box<Expr>, Box<Expr>),
    LowerName(String),
    NounBlock(Vec<Expr>, Box<Noun>),  // parenthesized

    Constant(Literal),

    // The underscore has the precedence of an adverb, but unlike other adverbs, it produces a
    // noun - that's the whole point - so it's treated specially here.
    Underscored(Box<SmallExpr>),

    // [a; b; c] or a b c
    ArrayLiteral(Vec<Elem>),

    // a[i1; i2;...; iN]
    // Represents indexing an array or calling a function.
    // Just like SmallVerb::NamedAdverbCall, but the result is treated as a noun. 
    // TODO should a[;j] be a verb?
    Indexed(Box<SmallNoun>, Vec<Elem>),
}

#[derive(Debug, Clone)]
pub enum Predicate {
    If2(Box<Expr>, Box<Expr>),  // Takes the condition as its left argument
    VerbCall(Verb, Option<SmallNoun>),
    ForwardAssignment(Pattern),
}

// TODO verb forward assignment?
#[derive(Debug, Clone)]
pub enum Verb {
    UpperAssign(String, Box<Verb>),
    SmallVerb(SmallVerb),
    Train(Box<SmallVerb>, Vec<TrainPart>),
}

#[derive(Debug, Clone)]
pub enum TrainPart {
    Fork(SmallVerb, SmallVerb),
    Atop(SmallVerb),
}

#[derive(Debug, Clone)]
pub enum Literal {
    Int(i64),
    Float(f64),

    // ( "a" ) is a character literal. Two ways to make a single-character
    // string are ( "a", ) and ( ["a"] ).
    Char(u8),
    String(String),
}

impl Literal {
    pub fn is_atom(&self) -> bool {
        use Literal::*;
        match self {
            Int(_) | Float(_) | Char(_) => true,
            String(_) => false,
        }
    }
}

// Part of an array literal or an argument list.
#[derive(Debug, Clone)]
pub enum Elem {
    // A single expression: one item of the array or argument list.
    Expr(Expr),

    // _ in an argument list of any length, or an absent item in an argument list with length >= 2.
    // e.g., in f[;3] the Elems are [MissingArg, Expr(3)]
    MissingArg,

    // A splicing of another array; this expands to 0 or more indvidual items. .. is parsed like an
    // adverb, but it can only be used in array literals and function arguments.
    Spliced(SmallExpr),
}

#[derive(Debug, Clone)]
pub enum SmallVerb {
    UpperName(String),
    VerbBlock(Vec<Expr>, Box<Verb>),  // parenthesized
    PrimVerb(PrimFunc),
    Lambda(Lambda),
    PrimAdverbCall(PrimAdverb, Box<SmallExpr>),
    PrimConjunctionCall(PrimConjunction, Box<SmallExpr>, Box<SmallExpr>),

    // Just like SmallNoun::Indexed, but the result is treated as a verb.
    NamedAdverbCall(Box<SmallVerb>, Vec<Elem>),
}

#[derive(Debug, Clone)]
pub enum Lambda {
    Short(Vec<Expr>),
    Cases(Vec<LambdaCase>),
}

#[derive(Debug, Clone)]
pub struct LambdaCase(pub ExplicitArgs, pub Vec<Expr>);

// Explicit argument syntax:
//   - Naming x: {[xPat] ...}
//   - Naming x and y: {[xPat; yPat] ...}
//
// Note that this brings about a syntactic weirdness where a lambda can't start immediately with a
// bracketed array literal; it can start with whitespace, parenthesize the array, or use a stranded
// array instead.
#[derive(Debug, Clone)]
pub struct ExplicitArgs(pub Vec<Option<Pattern>>);

#[derive(Debug, Clone)]
pub enum Pattern {
    Constant(Literal),

    // _
    Wildcard,

    Name(String),

    // Bracketed or stranded array
    Array(Vec<PatternElem>),

    // F->pat
    View(SmallVerb, Box<Pattern>),

    // Match two patterns at once, usually used to name the whole of a value
    // while also destructuring it, as in ( name->a b c ).
    As(Box<Pattern>, Box<Pattern>),
}

#[derive(Debug, Clone)]
pub enum PatternElem {
    Pattern(Pattern),

    // .. or ..name
    // Up to one subarray pattern is allowed in each Array pattern.
    Subarray(Option<String>),
}

#[derive(Debug, Clone)]
pub enum SmallExpr {
    Noun(SmallNoun),
    Verb(SmallVerb),
}

// `Elem` as it's allowed to appear in a stranded array literal. This isn't stored in the syntax
// tree, just used intermediately.
enum StrandedElem {
    Single(SmallNoun),
    Spliced(SmallExpr),
    Underscore,
}

enum Either<A, B> { Left(A), Right(B) }

impl StrandedElem {
    fn to_elem(self) -> Elem {
        match self {
            StrandedElem::Single(small_noun) => Elem::Expr(Expr::Noun(Noun::SmallNoun(small_noun))),
            StrandedElem::Spliced(small_expr) => Elem::Spliced(small_expr),
            StrandedElem::Underscore => Elem::MissingArg,
        }
    }
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

    fn consume(&mut self, tok: &Token) -> bool {
        if self.peek() == Some(tok) { self.skip(); true }
        else { false }
    }

    fn consume_or_fail(&mut self, tok: &Token) -> Res<()> {
        if self.consume(tok) {
            Ok(())
        } else {
            Err(self.expected(&format!("`{tok}'")))
        }
    }

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
        self.parse_sequenced_expecting(&"expression", Self::parse_expr)
    }

    fn parse_pattern_elems(&mut self) -> Many<PatternElem> {
        self.parse_sequenced_expecting(&"pattern", Self::parse_pattern_elem)
    }

    fn parse_elems(&mut self) -> Many<Elem> {
        self.parse_sequenced(|_| Ok(Elem::MissingArg), Self::parse_elem)
    }

    fn parse_patterns_allowing_empty(&mut self) -> Many<Option<Pattern>> {
        self.parse_sequenced(|_| Ok(None), |this| match this.parse_pattern() {
            Ok(Some(a)) => Ok(Some(Some(a))),
            Ok(None) => Ok(None),
            Err(e) => Err(e),
        })
    }

    fn parse_sequenced_expecting<A, F>(&mut self, expected_label: &str, parse: F) -> Many<A>
    where F: Fn(&mut Self) -> Parsed<A> {
        self.parse_sequenced(|this| Err(this.expected(expected_label)), parse)
    }

    fn parse_sequenced<A, F, Missing>(&mut self, missing: Missing, parse: F) -> Many<A>
    where F: Fn(&mut Self) -> Parsed<A>,
          Missing: Fn(&mut Self) -> Res<A> {
        let mut ret = Vec::with_capacity(2);  // arbitrary

        self.skip_newlines();
        let mut saw_semicolon = false;
        loop {
            if self.consume(&Token::Semicolon) {
                ret.push(missing(self)?);
                saw_semicolon = true;
            } else {
                match parse(self)? {
                    Some(parsed) => ret.push(parsed),
                    None => if saw_semicolon {
                        // There should have been something here.
                        ret.push(missing(self)?);
                    } else {
                        // There's nothing left, and we didn't see a semicolon at the end of the
                        // last iteration; must've just been some newlines.
                        break;
                    }
                }
                saw_semicolon = self.consume(&Token::Semicolon);
            };

            // We only want to continue if this parse is separated from the next by a semicolon
            // and/or newlines.
            if !saw_semicolon & !self.skip_newlines() { break; }
        }
        Ok(ret)
    }

    fn parse_expr(&mut self) -> Parsed<Expr> {
        let expr = if let Some(noun) = self.parse_noun()? {
            Expr::Noun(noun)
        } else if let Some(verb) = self.parse_verb()? {
            Expr::Verb(verb)
        } else if self.consume(&Token::Load) {
            match self.peek() {
                Some(Token::StrLit(mod_name)) => {
                    let mod_name = mod_name.clone();
                    self.skip();
                    Expr::PragmaLoad(mod_name)
                }
                _ => return Err(self.expected("string literal after `load'")),
            }
        } else {
            return Ok(None)
        };
        Ok(Some(expr))
    }

    fn parse_noun(&mut self) -> Parsed<Noun> {
        if self.consume(&Token::RightArrow) {
            let pattern = match self.parse_pattern()? {
                Some(pat) => pat,
                None => return Err(self.expected("pattern after `->'")),
            };
            let predicate = self.parse_predicates()?;
            return Ok(Some(Noun::ModifyingAssign(pattern, predicate)));
        }

        let before_small_noun = self.token_index;
        let small_noun = match self.parse_small_noun()? {
            Some(small_noun) => small_noun,
            None => return Ok(None),
        };

        let noun = match self.peek() {
            Some(Token::Colon) => {
                // Oh! Turns out `small_noun` wasn't a noun, it was a pattern.
                self.token_index = before_small_noun;
                let pattern = match self.parse_pattern()? {
                    Some(pattern) => pattern,
                    None => return Err(self.expected("pattern")),
                };
                assert!(self.consume(&Token::Colon));
                match self.parse_noun()? {
                    Some(rhs) => LowerAssign(pattern, Box::new(rhs)),
                    None => return Err(self.expected("RHS of noun assignment")),
                }
            }
            None => SmallNoun(small_noun),
            _ => Sentence(small_noun, self.parse_predicates()?),
        };

        Ok(Some(noun))
    }

    fn parse_splice(&mut self) -> Parsed<SmallExpr> {
        match self.peek() {
            Some(Token::DotDot {..}) => self.skip(),
            _ => return Ok(None)
        }
        match self.parse_small_expr(/*stranding_allowed=*/false)? {
            Some(spliced) => Ok(Some(spliced)),
            None => Err(self.expected(&format!("operand for `..'"))),
        }
    }

    fn parse_elem(&mut self) -> Parsed<Elem> {
        Ok(match self.parse_splice()? {
            Some(splice) => Some(Elem::Spliced(splice)),
            None => match self.peek() {
                Some(&Token::PrimAdverb(PrimAdverb::Underscore { before_whitespace })) => {
                    self.skip();
                    if before_whitespace {
                        Some(Elem::MissingArg)
                    } else {
                        match self.parse_small_expr(/*stranding_allowed=*/false)? {
                            None => Some(Elem::MissingArg),
                            Some(small_expr) => Some(Elem::Expr(Expr::Noun(
                                Noun::SmallNoun(SmallNoun::Underscored(Box::new(small_expr)))
                            ))),
                        }
                    }
                }
                _ => self.parse_expr()?.map(Elem::Expr),
            }
        })
    }

    fn parse_stranded_elem(&mut self) -> Parsed<StrandedElem> {
        Ok(match self.parse_splice()? {
            Some(splice) => Some(StrandedElem::Spliced(splice)),
            None => {
                let lone_underscore =
                    Token::PrimAdverb(PrimAdverb::Underscore { before_whitespace: true });
                if self.consume(&lone_underscore) {
                    Some(StrandedElem::Underscore)
                } else {
                    self.parse_small_noun_no_stranding()?.map(StrandedElem::Single)
                }
            }
        })
    }

    fn parse_small_noun(&mut self) -> Parsed<SmallNoun> {
        let stranded_elem = match self.parse_stranded_elem()? {
            Some(elem) => elem,
            None => return Ok(None),
        };
        match self.parse_stranded_elem()? {
            None => match stranded_elem {
                StrandedElem::Single(small_noun) => Ok(Some(small_noun)),
                StrandedElem::Spliced(_) => cold_err!("`..' is only allowed in array literals."),
                StrandedElem::Underscore =>
                    cold_err!("When not used as a pattern or argument placeholder, `_' \
                               must take an operand without intervening whitespace."),
            }
            Some(next_elem) => {
                let mut elems = vec![stranded_elem.to_elem(), next_elem.to_elem()];
                while let Some(next_elem) = self.parse_stranded_elem()? {
                    elems.push(next_elem.to_elem());
                }
                Ok(Some(ArrayLiteral(elems)))
            }
        }
    }

    fn parse_small_noun_no_stranding(&mut self) -> Parsed<SmallNoun> {
        // TODO prim nouns
        let mut small_noun = match self.peek() {
            Some(Token::C0Lower) => {
                self.skip();
                Constant(Literal::Char(0))
            }
            Some(&Token::IfLower) => {
                self.skip();
                self.consume_or_fail(&Token::LParen)?;
                match <[Expr; 3]>::try_from(self.parse_exprs()?) {
                    Ok([cond, then, else_]) => {
                        self.consume_or_fail(&Token::RParen)?;
                        Ok(If3(Box::new(cond), Box::new(then), Box::new(else_)))
                    }
                    Err(exprs) => if exprs.len() < 3 {
                        cold_err!("Not enough arguments to `if'; expected 3")
                    } else {
                        cold_err!("Too many arguments to `if'; expected 3")
                    }
                }?
            }
            Some(&Token::PrimAdverb(PrimAdverb::Underscore { before_whitespace })) => {
                if before_whitespace {
                    return Err(self.expected("operand for `_' with no intervening whitespace"));
                }
                self.skip();
                match self.parse_small_expr(/*stranding_allowed=*/false)? {
                    Some(operand) => Underscored(Box::new(operand)),
                    None => return Err(self.expected("operand for `_'")),
                }
            }
            Some(Token::LowerName(name)) => {
                let name_clone = name.clone();
                self.skip();
                LowerName(name_clone)
            }
            Some(&Token::IntLit(int)) => {
                self.skip();
                Constant(Literal::Int(int))
            }
            Some(&Token::FloatLit(float)) => {
                self.skip();
                Constant(Literal::Float(float))
            }
            Some(Token::StrLit(s)) => {
                let literal = if s.len() == 1 {
                    Literal::Char(s.as_bytes()[0])  // TODO unicode
                } else {
                    Literal::String(s.clone())
                };
                self.skip();
                Constant(literal)
            }
            Some(Token::LParen) => {
                // TODO backtracking can be inefficient if we have to go deep repeatedly. Instead we
                // can look ahead at the tokens of the last expression in the block, or refactor to
                // accept parsing a small noun *or* small verb here.
                let reset = self.token_index;
                self.skip();
                let mut exprs = self.parse_exprs()?;
                match exprs.pop() {
                    None => { todo!("Parse ()") }
                    Some(Expr::Noun(noun)) => {
                        self.consume_or_fail(&Token::RParen)?;
                        NounBlock(exprs, Box::new(noun))
                    }
                    _ => {
                        self.token_index = reset;
                        return Ok(None);
                    }
                }
            }
            Some(Token::LBracket{..}) => {
                self.skip();
                let elems = self.parse_elems()?;
                self.consume_or_fail(&Token::RBracket)?;
                ArrayLiteral(elems)
            }
            _ => return Ok(None),
        };

        while let Some(indices) = self.parse_bracketed_args()? {
            small_noun = Indexed(Box::new(small_noun), indices);
        }

        Ok(Some(small_noun))
    }

    fn parse_predicates(&mut self) -> Many<Predicate> {
        let mut predicates = vec![];
        loop {
            if self.consume(&Token::RightArrow) {
                match self.parse_pattern()? {
                    Some(pat) => predicates.push(Predicate::ForwardAssignment(pat)),
                    None => return Err(self.expected("pattern after `->'")),
                }
            } else if let Some(verb) = self.parse_small_verb()? {
                predicates.push(Predicate::VerbCall(Verb::SmallVerb(verb), self.parse_small_noun()?))
            } else if self.consume(&Token::IfUpper) {
                self.consume_or_fail(&Token::LParen)?;
                match <[Expr; 2]>::try_from(self.parse_exprs()?) {
                    Ok([then, else_]) => {
                        self.consume_or_fail(&Token::RParen)?;
                        predicates.push(Predicate::If2(Box::new(then), Box::new(else_)));
                    }
                    Err(exprs) => return if exprs.len() < 2 {
                        cold_err!("Not enough arguments to `if'; expected 2")
                    } else {
                        cold_err!("Too many arguments to `if'; expected 2")
                    }
                }
            } else {
                break
            }
        }
        Ok(predicates)
    }

    fn parse_verb(&mut self) -> Parsed<Verb> {
        let small_verb = match self.parse_small_verb()? {
            Some(small_verb) => match self.parse_small_noun()? {
                Some(bound_arg) => NamedAdverbCall(
                    Box::new(small_verb),
                    vec![Elem::MissingArg, Elem::Expr(Expr::Noun(Noun::SmallNoun(bound_arg)))]
                ),
                _ => small_verb,
            }
            None => return Ok(None),
        };

        if self.consume(&Token::Colon) {
            return match small_verb {
                UpperName(name) => match self.parse_verb()? {
                    Some(rhs) => Ok(Some(UpperAssign(name, Box::new(rhs)))),
                    None => Err(self.expected("RHS of verb assignment")),
                }
                _ => cold_err!("Invalid verb assignment target: {small_verb:?}"),
            }
        }

        let mut train = vec![];
        while let Some(next_verb) = self.parse_small_verb()? {
            let train_part = match self.parse_small_verb()? {
                Some(verb_rhs) => TrainPart::Fork(next_verb, verb_rhs),
                None => TrainPart::Atop(
                    match self.parse_small_noun()? {
                        Some(noun_rhs) => NamedAdverbCall(
                            Box::new(next_verb),
                            vec![Elem::MissingArg, Elem::Expr(Expr::Noun(Noun::SmallNoun(noun_rhs)))]
                        ),
                        None => next_verb,
                    }
                ),
            };
            train.push(train_part);
        }

        Ok(Some(if !train.is_empty() {
            Verb::Train(Box::new(small_verb), train)
        } else {
            Verb::SmallVerb(small_verb)
        }))
    }

    fn parse_small_pattern(&mut self) -> Parsed<Pattern> {
        let pattern = match self.peek() {
            Some(&Token::IntLit(int)) => {
                self.skip();
                Pattern::Constant(Literal::Int(int))
            }
            Some(&Token::FloatLit(float)) => {
                self.skip();
                Pattern::Constant(Literal::Float(float))
            }
            Some(Token::StrLit(s)) => {
                let literal = if s.len() == 1 {
                    Literal::Char(s.as_bytes()[0])
                } else {
                    Literal::String(s.clone())
                };
                self.skip();
                Pattern::Constant(literal)
            }
            Some(Token::LowerName(name)) => {
                let pat = Pattern::Name(name.clone());
                self.skip();
                pat
            }
            Some(&Token::PrimAdverb(PrimAdverb::Underscore { .. })) => {
                self.skip();
                Pattern::Wildcard
            }
            Some(Token::LBracket{..}) => {
                self.skip();
                let elems = self.parse_pattern_elems()?;
                self.consume_or_fail(&Token::RBracket)?;
                Pattern::Array(elems)
            }
            Some(Token::LParen) => {
                self.skip();
                let inner = match self.parse_pattern()? {
                    Some(pat) => pat,
                    None => return Err(self.expected("pattern")),
                };
                self.consume_or_fail(&Token::RParen)?;
                inner
            }
            _ => return Ok(None),
        };
        Ok(Some(pattern))
    }

    fn parse_small_pattern_elem(&mut self) -> Parsed<PatternElem> {
        match self.parse_subarray_pattern()? {
            Some(pat) => Ok(Some(pat)),
            None => Ok(self.parse_small_pattern()?.map(PatternElem::Pattern)),
        }
    }

    fn parse_subarray_pattern(&mut self) -> Parsed<PatternElem> {
        let whitespace_after_dotdot = match self.peek() {
            Some(&Token::DotDot { before_whitespace }) => {
                self.skip();
                before_whitespace
            }
            _ => return Ok(None),
        };

        let elem = match self.peek() {
            Some(Token::LowerName(name)) if !whitespace_after_dotdot => {
                let name = name.clone();
                self.skip();
                PatternElem::Subarray(Some(name))
            }
            _ => PatternElem::Subarray(None),
        };
        Ok(Some(elem))
    }

    fn parse_pattern(&mut self) -> Parsed<Pattern> {
        match self.parse_pattern_elem()? {
            Some(PatternElem::Pattern(pat)) => Ok(Some(pat)),
            Some(PatternElem::Subarray(_)) => cold_err!("`..' is only allowed in array patterns"),
            None => Ok(None),
        }
    }

    fn parse_as_pattern(&mut self) -> Parsed<Pattern> {
        if !self.consume(&Token::RightArrow) { return Ok(None); }
        let arrow_rhs = self.parse_pattern()?;
        if arrow_rhs.is_none() {
            return Err(self.expected("pattern after `->'"));
        }
        Ok(arrow_rhs)
    }

    // If both are Err, returns the error of whichever one made more progress in the token stream
    // before failing. If they made the same progress, we choose the error returned by `parse_left`.
    // 
    // TODO: combine the errors instead.
    fn either<A, B>(
        &mut self,
        parse_left: impl FnOnce(&mut Self) -> Parsed<A>,
        parse_right: impl FnOnce(&mut Self) -> Parsed<B>
    ) -> Parsed<Either<A, B>> {
        let backtrack = self.token_index;
        match parse_left(self) {
            Ok(Some(left)) => Ok(Some(Either::Left(left))),
            Ok(None) => {
                self.token_index = backtrack;
                match parse_right(self)? {
                    Some(right) => Ok(Some(Either::Right(right))),
                    None => Ok(None),
                }
            }
            Err(left_err) => {
                let index_of_unexpected_in_left = self.token_index;
                self.token_index = backtrack;
                match parse_right(self) {
                    Ok(Some(right)) => Ok(Some(Either::Right(right))),
                    Ok(None) => Err(left_err),
                    Err(right_err) if self.token_index > index_of_unexpected_in_left => Err(right_err),
                    _ => {
                        self.token_index = index_of_unexpected_in_left;
                        return Err(left_err);
                    }
                }
            }
        }
    }

    fn parse_pattern_elem(&mut self) -> Parsed<PatternElem> {
        fn view_pattern(verb: Verb, arrow_rhs: Pattern) -> Pattern {
            Pattern::View(VerbBlock(vec![], Box::new(verb)), Box::new(arrow_rhs))
        }

        let before_lparen = self.token_index;
        let nested_or_view_pat = if self.consume(&Token::LParen) {
            // This is either (pattern) or (verb)->pattern. The only way to find out is to try them
            // both; we need to be careful because we may parse quite a bit past the LParen before
            // finding out.
            match self.either(|this| this.parse_pattern(), |this| this.parse_verb())? {
                Some(pattern_or_verb) => {
                    self.consume_or_fail(&Token::RParen)?;
                    match pattern_or_verb {
                        Either::Left(pat) => Some(pat),
                        Either::Right(verb) => match self.parse_as_pattern()? {
                            Some(arrow_rhs) => Some(view_pattern(verb, arrow_rhs)),
                            None => return Err(self.expected("`->")),
                        }
                    }
                }
                None => {
                    self.token_index = before_lparen;
                    None
                }
            }
        } else if let Some(small_verb) = self.parse_small_verb()? {
            match self.parse_as_pattern()? {
                Some(arrow_rhs) => Some(Pattern::View(small_verb, Box::new(arrow_rhs))),
                None => return Err(self.expected("`->'")),
            }
        } else {
            None
        };

        let elem = match nested_or_view_pat {
            Some(pat) => PatternElem::Pattern(pat),
            None => match self.parse_small_pattern_elem()? {
                Some(small_pat) => small_pat,
                None => return Ok(None),
            }
        };

        // Stranding
        let mut pat = match self.parse_small_pattern_elem()? {
            None => match elem {
                PatternElem::Subarray(_) => return Ok(Some(elem)),
                PatternElem::Pattern(pat) => pat
            }

            Some(next_elem) => {
                let mut stranded_pats = vec![elem, next_elem];
                while let Some(next_elem) = self.parse_small_pattern_elem()? {
                    stranded_pats.push(next_elem)
                }
                let num_subarray_patterns = stranded_pats.iter()
                    .filter(|elem| matches!(elem, PatternElem::Subarray(_)))
                    .count();
                if num_subarray_patterns > 1 {
                    return cold_err!("Only one `..' is allowed per array-matching pattern.");
                }
                Pattern::Array(stranded_pats)
            }
        };

        // As-patterns
        while let Some(arrow_rhs) = self.parse_as_pattern()? {
            pat = Pattern::As(Box::new(pat), Box::new(arrow_rhs));
        }

        Ok(Some(PatternElem::Pattern(pat)))
    }

    fn parse_explicit_args(&mut self) -> Parsed<ExplicitArgs> {
        self.skip_newlines();

        // Allow an optional leading colon for uniformity with multiple cases.
        let initial_colon = self.consume(&Token::Colon) || self.consume(&Token::ColonAfterWhitespace);
        self.skip_newlines();
        let args_present = self.consume(&Token::LBracket { after_whitespace: false }) ||
                           initial_colon && self.consume(&Token::LBracket { after_whitespace: true });
        if !args_present {
            if initial_colon {
                return Err(self.expected("`[' and explicit arguments after `:'"));
            } else {
                return Ok(None)
            }
        }

        self.skip_newlines();
        let patterns = self.parse_patterns_allowing_empty()?;
        self.consume_or_fail(&Token::RBracket)?;

        Ok(Some(ExplicitArgs(patterns)))
    }

    fn parse_small_verb(&mut self) -> Parsed<SmallVerb> {
        let mut small_verb = match self.peek() {
            Some(Token::C0Upper) => {
                self.skip();
                PrimAdverbCall(
                    PrimAdverb::Dot,
                    Box::new(SmallExpr::Noun(SmallNoun::Constant(Literal::Char(0))))
                )
            }
            Some(Token::UpperName(name)) => {
                let upper_name = name.clone();
                self.skip();
                UpperName(upper_name)
            }
            Some(Token::LBrace) => {
                self.skip();
                let lambda = match self.parse_explicit_args()? {
                    None => {
                        let exprs = self.parse_exprs()?;
                        if exprs.is_empty() {  // TODO remove to enable {} or {[args]} parse
                            return Err(self.expected("expression"))
                        }
                        Lambda::Short(exprs)
                    }
                    Some(mut explicit_args) => {
                        let mut cases = vec![];
                        loop {
                            let exprs = self.parse_exprs()?;
                            if exprs.is_empty() {  // TODO remove to enable {} or {[args]} parse
                                return Err(self.expected("expression"))
                            }
                            cases.push(LambdaCase(explicit_args, exprs));
                            if self.peek() != Some(&Token::ColonAfterWhitespace) { break }
                            // parse_explicit_args is guaranteed to consume the colon.
                            explicit_args = self.parse_explicit_args()?.unwrap();
                        }
                        Lambda::Cases(cases)
                    }
                };

                if !self.consume(&Token::RBrace) {
                    return Err(self.expected("`;', newline, or `}}'"))
                }
                Lambda(lambda)
            }
            Some(&Token::PrimVerb(verb)) => {
                self.skip();
                PrimVerb(PrimFunc::Verb(verb))
            }
            // Don't capture `_' here: the underscore produces a noun, and we're looking for a verb.
            Some(&Token::PrimAdverb(adverb)) if !matches!(adverb, PrimAdverb::Underscore {..})  => {
                self.skip();
                // TODO: we have a problem; x .(func...) y parses as x .((func...) y), a monadic
                // invocation of a .-derived verb whose operand is a stranded array of two elements
                match self.parse_small_expr(/*stranding_allowed=*/false)? {
                    Some(operand) => PrimAdverbCall(adverb, Box::new(operand)),
                    None => return Err(self.expected(&format!("operand for adverb `{adverb}'"))),
                }
            }
            Some(Token::LParen) => {
                // TODO backtracking can be inefficient if we have to go deep repeatedly. Instead we
                // can look ahead at the tokens of the last expression in the block, or refactor to
                // accept parsing a small noun *or* small verb here.
                let reset = self.token_index;
                self.skip();
                let mut exprs = self.parse_exprs()?;
                match exprs.pop() {
                    Some(Expr::Verb(verb)) => {
                        self.consume_or_fail(&Token::RParen)?;
                        VerbBlock(exprs, Box::new(verb))
                    }
                    _ => {
                        self.token_index = reset;
                        return Ok(None);
                    }
                }
            }
            _ => return Ok(None),
        };

        if matches!(small_verb, UpperName(_) | Lambda(_) | VerbBlock(_, _)) {
            while let Some(args) = self.parse_bracketed_args()? {
                small_verb = NamedAdverbCall(Box::new(small_verb), args);
            }
        }

        if let Some(&Token::PrimConjunction(prim_conj)) = self.peek() {
            self.skip();
            small_verb = match self.parse_small_expr(/*stranding_allowed=*/false)? {
                Some(right_operand) => PrimConjunctionCall(
                    prim_conj, Box::new(SmallExpr::Verb(small_verb)), Box::new(right_operand)
                ),
                None => return Err(self.expected(&format!("right operand for conjunction `{prim_conj}'"))),
            };
        }

        Ok(Some(small_verb))
    }

    fn parse_bracketed_args(&mut self) -> Parsed<Vec<Elem>> {
        if self.consume(&Token::LBracket{after_whitespace: false}) {
            let args = self.parse_elems()?;
            self.consume_or_fail(&Token::RBracket)?;
            Ok(Some(args))
        } else {
            Ok(None)
        }
    }

    fn parse_small_expr(&mut self, stranding_allowed: bool) -> Parsed<SmallExpr> {
        let small_noun = if stranding_allowed {
            self.parse_small_noun()
        } else {
            self.parse_small_noun_no_stranding()
        }?;

        let small_expr = if let Some(small_noun) = small_noun {
            SmallExpr::Noun(small_noun)
        } else if let Some(small_verb) = self.parse_small_verb()? {
            SmallExpr::Verb(small_verb)
        } else {
            return Ok(None)
        };
        Ok(Some(small_expr))
    }
}
