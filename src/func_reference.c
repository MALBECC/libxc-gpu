/*
 Copyright (C) 20017 M. Oliveira

 This Source Code Form is subject to the terms of the Mozilla Public
 License, v. 2.0. If a copy of the MPL was not distributed with this
 file, You can obtain one at http://mozilla.org/MPL/2.0/.
*/

#include <xc.h>


char const *xc_func_reference_get_ref_cuda(const func_reference_type_cuda *reference)
{
    return reference->ref;
}

char const *xc_func_reference_get_doi_cuda(const func_reference_type_cuda *reference)
{
    return reference->doi;
}

char const *xc_func_reference_get_bibtex_cuda(const func_reference_type_cuda *reference)
{
    return reference->bibtex;
}
