#![cfg_attr(not(test), warn(unused_crate_dependencies))]

use proc_macro::TokenStream;
use proc_macro2::{TokenStream as TokenStream2, TokenTree};
use quote::quote;
use syn::{braced, parse::Parse, parse_macro_input, Expr, Ident, Token};

const MIN_UNROLL: usize = 1;
const MAX_UNROLL: usize = 5;

struct UnrollInput {
    n_expr: Expr,
    placeholder: Ident,
    body: TokenStream2,
}

impl Parse for UnrollInput {
    fn parse(input: syn::parse::ParseStream) -> syn::Result<Self> {
        let n_expr: Expr = input.parse()?;
        input.parse::<Token![,]>()?;
        let placeholder: Ident = input.parse()?;
        input.parse::<Token![,]>()?;
        let body_content;
        braced!(body_content in input);
        let body: TokenStream2 = body_content.parse()?;

        Ok(UnrollInput {
            n_expr,
            placeholder,
            body,
        })
    }
}

fn replace_placeholder(tokens: TokenStream2, placeholder: &Ident, value: usize) -> TokenStream2 {
    tokens
        .into_iter()
        .map(|tt| match tt {
            TokenTree::Ident(ref ident) if ident == placeholder => {
                TokenTree::Literal(proc_macro2::Literal::usize_unsuffixed(value))
            }
            TokenTree::Group(group) => {
                let new_stream = replace_placeholder(group.stream(), placeholder, value);
                TokenTree::Group(proc_macro2::Group::new(group.delimiter(), new_stream))
            }
            other => other,
        })
        .collect()
}

/// Generates a match expression that unrolls loop iterations with const generic indices.
///
/// # Syntax
/// ```ignore
/// unroll_match!(MAX, expr, PLACEHOLDER, { body });
/// ```
///
/// - `MAX`: Maximum number of iterations (literal)
/// - `expr`: Expression to match on
/// - `PLACEHOLDER`: Identifier to replace with 0, 1, 2, ... in each iteration
/// - `body`: Code block containing the placeholder
#[proc_macro]
pub fn unroll_match(input: TokenStream) -> TokenStream {
    let UnrollInput {
        n_expr,
        placeholder,
        body,
    } = parse_macro_input!(input as UnrollInput);

    let arms: Vec<TokenStream2> = (MIN_UNROLL..=MAX_UNROLL)
        .map(|n| {
            let statements: Vec<TokenStream2> = (0..n)
                .map(|i| replace_placeholder(body.clone(), &placeholder, i))
                .collect();

            let n_lit = proc_macro2::Literal::usize_unsuffixed(n);
            quote! {
                #n_lit => {
                    #(#statements)*
                }
            }
        })
        .collect();

    let expanded = quote! {
        match #n_expr {
            #(#arms)*
            _ => unreachable!()
        }
    };

    expanded.into()
}