package com.sicklescan.app

/**
 * Pure presentation rules for the result screen (no Android dependency, so they
 * are unit-tested locally).
 */
object ResultsPresentation {

    /**
     * The disease models are only meaningful on their own image type (e.g. the
     * sickle-cell model outputs a confident "positive" on malaria cell images), so
     * whenever results for two conditions are shown together the cross-domain caution
     * is shown with them -- on every such result screen, not just the first. It is a
     * visibility aid only: "Both" is never blocked or disabled.
     */
    fun showCrossDomainWarning(shownDiseases: Collection<Disease>): Boolean =
        shownDiseases.toSet().size >= 2
}
