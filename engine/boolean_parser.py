"""
boolean_parser.py
Parses Boolean query strings into an abstract syntax tree (AST).

Supports: AND, OR, NOT, parentheses, and bare terms.
Example: "covid AND (treatment OR vaccine) AND NOT influenza"

The AST is a nested tuple structure:
  ("AND", [child1, child2, ...])
  ("OR",  [child1, child2, ...])
  ("NOT", child)
  ("TERM", "stemmed_word")
"""

import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline.tokenizer import stem_term


class ParseError(Exception):
    """Raised when the query string cannot be parsed."""
    pass


def tokenize_query(query_str):
    """
    Splits a Boolean query string into tokens: words, operators, parentheses.
    Operators AND, OR, NOT are case-insensitive.
    Wildcards (terms ending with *) are preserved as-is.
    Returns a list of token strings.
    """
    # regex: match words (possibly with * for wildcard), or single-char parens
    tokens = re.findall(r"[a-zA-Z0-9*]+|[()]", query_str)
    # normalize operators to uppercase
    normalized = []
    for t in tokens:
        upper = t.upper()
        if upper in ("AND", "OR", "NOT"):
            normalized.append(upper)
        else:
            normalized.append(t)
    return normalized


def parse(query_str):
    """
    Parses a Boolean query string into an AST.
    Grammar (simplified):
      expr     -> or_expr
      or_expr  -> and_expr (OR and_expr)*
      and_expr -> not_expr (AND not_expr)*
      not_expr -> NOT not_expr | atom
      atom     -> TERM | '(' expr ')'

    If no operators are present, terms are implicitly ANDed.
    """
    tokens = tokenize_query(query_str)
    if not tokens:
        raise ParseError("Empty query")

    # check if any operators are present; if not, implicit AND
    has_operators = any(t in ("AND", "OR", "NOT") or t in ("(", ")") for t in tokens)
    if not has_operators:
        # treat "covid treatment vaccine" as "covid AND treatment AND vaccine"
        terms = [_make_term_node(t) for t in tokens]
        if len(terms) == 1:
            return terms[0]
        return ("AND", terms)

    pos = [0]  # mutable position counter (list so inner functions can modify it)

    def peek():
        if pos[0] < len(tokens):
            return tokens[pos[0]]
        return None

    def consume():
        t = tokens[pos[0]]
        pos[0] += 1
        return t

    def parse_or():
        """Parse OR expressions (lowest precedence)."""
        left = parse_and()
        children = [left]
        while peek() == "OR":
            consume()  # eat "OR"
            children.append(parse_and())
        if len(children) == 1:
            return children[0]
        return ("OR", children)

    def parse_and():
        """Parse AND expressions (higher precedence than OR)."""
        left = parse_not()
        children = [left]
        while peek() is not None and peek() not in ("OR", ")"):
            if peek() == "AND":
                consume()
            children.append(parse_not())
        if len(children) == 1:
            return children[0]
        return ("AND", children)

    def parse_not():
        """Parse NOT (prefix unary operator)."""
        if peek() == "NOT":
            consume()  # eat "NOT"
            child = parse_not()  # NOT is right-associative
            return ("NOT", child)
        return parse_atom()

    def parse_atom():
        """Parse a term or parenthesized sub-expression."""
        t = peek()
        if t is None:
            raise ParseError("Unexpected end of query")
        if t == "(":
            consume()  # eat "("
            node = parse_or()
            if peek() != ")":
                raise ParseError("Missing closing parenthesis")
            consume()  # eat ")"
            return node
        if t in ("AND", "OR", ")"):
            raise ParseError(f"Unexpected token: {t}")
        consume()
        return _make_term_node(t)

    result = parse_or()

    # check that we consumed all tokens
    if pos[0] < len(tokens):
        raise ParseError(f"Unexpected token at position {pos[0]}: {tokens[pos[0]]}")

    return result


def _make_term_node(token):
    """
    Creates a TERM or WILDCARD node from a raw token.
    If the token ends with *, it is a wildcard query.
    Otherwise, stem it and create a TERM node.
    """
    if token.endswith("*"):
        # wildcard: keep the prefix (without *), stem it
        prefix = token[:-1].lower()
        return ("WILDCARD", prefix)
    else:
        return ("TERM", stem_term(token))


def ast_to_string(node):
    """
    Converts an AST back to a readable Boolean query string.
    Useful for logging what the engine is actually executing.
    """
    op = node[0]
    if op == "TERM":
        return node[1]
    elif op == "WILDCARD":
        return node[1] + "*"
    elif op == "NOT":
        child_str = ast_to_string(node[1])
        return f"NOT {child_str}"
    elif op in ("AND", "OR"):
        children = node[1]
        parts = [ast_to_string(c) for c in children]
        joiner = f" {op} "
        result = joiner.join(parts)
        return f"({result})"
    return str(node)


def collect_terms(node):
    """
    Collects all TERM values from an AST.
    Returns a set of stemmed terms (excludes wildcards and NOTs).
    """
    op = node[0]
    if op == "TERM":
        return {node[1]}
    elif op == "WILDCARD":
        return set()
    elif op == "NOT":
        return set()  # terms under NOT are excluded, not searched for positively
    elif op in ("AND", "OR"):
        result = set()
        for child in node[1]:
            result |= collect_terms(child)
        return result
    return set()


if __name__ == "__main__":
    # quick test
    tests = [
        "covid AND treatment",
        "covid OR coronavirus",
        "covid AND NOT influenza",
        "covid AND (treatment OR vaccine)",
        "treat*",
        "coronavirus origin transmission",
    ]
    for q in tests:
        try:
            ast = parse(q)
            print(f"Query: {q}")
            print(f"  AST: {ast}")
            print(f"  Str: {ast_to_string(ast)}")
            print(f"  Terms: {collect_terms(ast)}")
            print()
        except ParseError as e:
            print(f"Query: {q} -> ERROR: {e}")
