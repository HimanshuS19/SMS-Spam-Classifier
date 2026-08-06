/*
==========================================================
SpamShield AI: Intelligent SMS Spam Detection System
Frontend JavaScript
Author : Himanshu Singh
==========================================================
*/

document.addEventListener("DOMContentLoaded", () => {

    // ======================================================
    // Elements
    // ======================================================

    const textarea = document.getElementById("message");
    const counter = document.getElementById("charCount");
    const form = document.querySelector("form");
    const button = document.querySelector("button");

    // ======================================================
    // Character Counter
    // ======================================================

    function updateCounter() {

        if (!textarea || !counter) return;

        counter.textContent =
            `${textarea.value.length} / 1000`;

    }

    // ======================================================
    // Auto Resize Textarea
    // ======================================================

    function autoResize() {

        if (!textarea) return;

        textarea.style.height = "auto";
        textarea.style.height =
            textarea.scrollHeight + "px";

    }

    // ======================================================
    // Form Loading Animation
    // ======================================================

    function showLoading() {

        if (!button) return;

        button.disabled = true;

        button.innerHTML = `
            <i class="fa-solid fa-spinner fa-spin"></i>
            Analyzing...
        `;
    }

    // ======================================================
    // Animate Progress Bars
    // ======================================================

    function animateBars() {

        const bars =
            document.querySelectorAll(".progress-fill");

        bars.forEach(bar => {

            const width = bar.style.width;

            bar.style.width = "0%";

            setTimeout(() => {

                bar.style.width = width;

            }, 200);

        });

    }

    // ======================================================
    // Fade Result Card
    // ======================================================

    function animateResultCard() {

        const card =
            document.querySelector(".result-card");

        if (!card) return;

        card.style.opacity = "0";
        card.style.transform = "translateY(40px)";

        setTimeout(() => {

            card.style.transition =
                "all .8s ease";

            card.style.opacity = "1";
            card.style.transform =
                "translateY(0)";

        }, 100);

    }

    // ======================================================
    // Prevent Empty Submission
    // ======================================================

    if (form) {

        form.addEventListener("submit", function (e) {

            if (textarea.value.trim() === "") {

                e.preventDefault();

                textarea.focus();

                return;

            }

            showLoading();

        });

    }

    // ======================================================
    // Initialize
    // ======================================================

    if (textarea) {

        updateCounter();
        autoResize();

        textarea.addEventListener(
            "input",
            updateCounter
        );

        textarea.addEventListener(
            "input",
            autoResize
        );

    }

    animateBars();
    animateResultCard();

});