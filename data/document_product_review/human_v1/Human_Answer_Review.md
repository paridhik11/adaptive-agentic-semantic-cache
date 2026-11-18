# Human answer review

These questions were written by a model. Its judgments are hidden here.
Your review is human review of model-authored questions, not human-authored wording.

For each question judge the supplied complete cached answer: **Y** = directly answers every requirement; **N** = misses or contradicts a requirement; **?** = uncertain.
Related information alone is insufficient. Judge the answer as written; do not mentally add missing code or facts.
Reply in chat, e.g. `H01 Y, H02 N — missing requirement, H03 ?`. You can send one section at a time.
Add a short reason for N or ?. Do not inspect the separate model review until finished.

## How do I specify hexadecimal and octal integers?

Cached answer:
```text
To specify an octal digit, precede the octal value with a zero, and then a lower
or uppercase "o".  For example, to set the variable "a" to the octal value "10"
(8 in decimal), type::

   >>> a = 0o10
   >>> a
   8

Hexadecimal is just as easy.  Simply precede the hexadecimal number with a zero,
and then a lower or uppercase "x".  Hexadecimal digits can be specified in lower
or uppercase.  For example, in the Python interpreter::

   >>> a = 0xa5
   >>> a
   165
   >>> b = 0XB2
   >>> b
   178
```

**H01.** How do I write octal, hexadecimal, and binary integer literals in Python?

Your judgment: ___

**H02.** How do I turn the decimal integer 165 into a string containing its hexadecimal representation?

Your judgment: ___

**H03.** How do I write octal and hexadecimal integer literals in Python?

Your judgment: ___

**H04.** How do I write octal 10 and hexadecimal a5, then add them and print the total in octal?

Your judgment: ___

**H05.** How can I enter octal ten and hexadecimal a5 in the Python interpreter, and what values do they have?

Your judgment: ___

**H06.** How do I write hexadecimal floating-point literals in Python?

Your judgment: ___

**H07.** How do I parse the strings "a5" and "10" as hexadecimal and octal integers at runtime?

Your judgment: ___

**H08.** What prefixes do I put before base-eight and base-sixteen numbers in Python code?

Your judgment: ___

**H09.** When entering an integer in octal or hex, are uppercase prefix letters allowed?

Your judgment: ___

**H10.** Can you show how to assign a hexadecimal integer and an octal integer to a variable?

Your judgment: ___

## How do I convert between tuples and lists?

Cached answer:
```text
The type constructor tuple(seq) converts any sequence (actually, any
iterable) into a tuple with the same items in the same order.

For example, tuple([1, 2, 3]) yields (1, 2, 3) and tuple('abc')
yields ('a', 'b', 'c').  If the argument is a tuple, it does not make a copy
but returns the same object, so it is cheap to call tuple when you
aren't sure that an object is already a tuple.

The type constructor list(seq) converts any sequence or iterable into a list
with the same items in the same order.  For example, list((1, 2, 3)) yields
[1, 2, 3] and list('abc') yields ['a', 'b', 'c'].  If the argument
is a list, it makes a copy just like seq[:] would.
```

**H11.** How do I convert a list to a tuple while making a deep copy of every nested object?

Your judgment: ___

**H12.** How do I convert a list of lists into a tuple of tuples?

Your judgment: ___

**H13.** Which constructors let me switch between lists and tuples while keeping the elements in order?

Your judgment: ___

**H14.** How do I convert a list into a tuple with the elements in reverse order?

Your judgment: ___

**H15.** Can list() and tuple() accept an iterable, and do they preserve its item order?

Your judgment: ___

**H16.** How do I convert a tuple to a list in place so existing references point to that list?

Your judgment: ___

**H17.** When I call tuple() on a tuple or list() on a list, do I get a copy?

Your judgment: ___

**H18.** How can I convert a tuple into a set and remove duplicate values?

Your judgment: ___

**H19.** How do I convert a tuple into a list and a list into a tuple in Python?

Your judgment: ___

**H20.** What do tuple([1, 2, 3]) and list((1, 2, 3)) produce?

Your judgment: ___

## What's a negative index?

Cached answer:
```text
Python sequences are indexed with positive numbers and negative numbers.  For
positive numbers 0 is the first index 1 is the second index and so forth.  For
negative indices -1 is the last index and -2 is the penultimate (next to last)
index and so forth.  Think of seq[-n] as the same as seq[len(seq)-n].

Using negative indices can be very convenient.  For example S[:-1] is all of
the string except for its last character, which is useful for removing the
trailing newline from a string.
```

**H21.** How can I get the last character as a one-character slice rather than as an indexed element?

Your judgment: ___

**H22.** What happens if a negative index is farther from the end than the sequence length, and does a slice behave the same way?

Your judgment: ___

**H23.** Does S[:-1] remove the last character from S itself, and how can I update S with the result?

Your judgment: ___

**H24.** Can you explain zero-based positive indexing and how negative indexing counts from the end?

Your judgment: ___

**H25.** How do -1 and -2 work when indexing a list or string?

Your judgment: ___

**H26.** What does a negative index mean for a Python sequence?

Your judgment: ___

**H27.** How do I use one-based indexing so index 1 retrieves the first item in a Python sequence?

Your judgment: ___

**H28.** How do I reverse a sequence using a slice with a negative step?

Your judgment: ___

**H29.** How can I express seq[-n] using the length of seq?

Your judgment: ___

**H30.** What does S[:-1] select from a string?

Your judgment: ___
