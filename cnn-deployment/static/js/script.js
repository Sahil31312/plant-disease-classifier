// Language and Direction Switching with AJAX
$(document).ready(function() {
    // Update page content function
    function updatePageContent(lang, direction) {
        $.ajax({
            url: '/api/page_content',
            method: 'GET', 
            data: { lang: lang },
            success: function(response) {
                if (response.success) {
                    // Update page title
                    document.title = response.title;

                    // Update all elements with data attributes
                    updateTranslations(response.translations);
                }
            }
        });
    }

    // Update translations
    function updateTranslations(translations) {
        // Update data-translate elements
        $('[data-translate]').each(function() {
            const key = $(this).data('translate');
            if (translations[key]) {
                $(this).text(translations[key]);
            }
        });

        // Update data-translate-placeholder elements
        $('[data-translate-placeholder]').each(function() {
            const key = $(this).data('translate-placeholder');
            if (translations[key]) {
                $(this).attr('placeholder', translations[key]);
            }
        });

        // Update data-translate-title elements
        $('[data-translate-title]').each(function() {
            const key = $(this).data('translate-title');
            if (translations[key]) {
                $(this).attr('title', translations[key]);
            }
        });
    }

    // Toast notification function
    function showToast(message, type = 'info') {
        const toastId = 'toast-' + Date.now();
        const direction = $('html').attr('dir') || 'ltr';
        const position = direction === 'rtl' ? 'left' : 'right';

        const toastHtml = `
            <div id="${toastId}" class="toast align-items-center text-bg-${type} border-0 position-fixed" 
                 role="alert" aria-live="assertive" aria-atomic="true" 
                 style="top: 20px; ${position}: 20px; z-index: 9999;">
                <div class="d-flex">
                    <div class="toast-body">
                        ${message}
                    </div>
                    <button type="button" class="btn-close btn-close-white me-2 m-auto" data-bs-dismiss="toast"></button>
                </div>
            </div>
        `;

        $('body').append(toastHtml);

        const toastElement = new bootstrap.Toast(document.getElementById(toastId));
        toastElement.show();

        $(`#${toastId}`).on('hidden.bs.toast', function() {
            $(this).remove();
        });
    }

    // Form validation for contact form
    $('#contactForm').submit(function(e) {
        const email = $('#email').val();
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;

        if (!emailRegex.test(email)) {
            e.preventDefault();
            const lang = $('html').attr('lang') || 'en';
            const errorMsg = lang === 'ps' ? 'مهرباني وکړئ یو باوري بریښنالیک ولیکئ' : 'Please enter a valid email address';
            showToast(errorMsg, 'error');
            return false;
        }

        $('#submitBtn').prop('disabled', true).html('<i class="fas fa-spinner fa-spin me-2"></i>Sending...');
        return true;
    });

    // File upload preview
    $('#fileInput').change(function() {
        const file = this.files[0];
        if (file) {
            const reader = new FileReader();
            reader.onload = function(e) {
                $('#previewImage').attr('src', e.target.result).show();
                $('#imagePreview').hide();
            }
            reader.readAsDataURL(file);
        }
    });

    // Initialize Bootstrap components
    $('[data-bs-toggle="tooltip"]').tooltip();
    $('[data-bs-toggle="popover"]').popover();

    // Auto-hide alerts after 5 seconds
    setTimeout(function() {
        $('.alert').alert('close');
    }, 5000);
});

//   language switching
$(document).ready(function() {
    $('.lang-btn').click(function(e) {
        e.preventDefault();
        const lang = $(this).data('lang');

        // Set language via AJAX
        $.ajax({
            url: '/set_lang/' + lang,
            method: 'GET',
            success: function() {
                // Reload the page to apply translations
                location.reload();
            },
            error: function() {
                showToast('Error switching language', 'error');
            }
        });
    });
 $('.lang-btn').click(function(e) {
        e.preventDefault();
        const lang = $(this).data('lang');

        // Set language via AJAX
        $.ajax({
            url: '/set_lang/' + lang,
            method: 'GET',
            success: function() {
                // Reload the page to apply translations
                location.reload();
            },
            error: function() {
                showToast('Error switching language', 'error');
            }
        });
    });

    // Initialize active button
    const currentLang = $('html').attr('lang') || 'en';
    $('.lang-btn').removeClass('active');
    $('.lang-btn[data-lang="' + currentLang + '"]').addClass('active');
 
});
