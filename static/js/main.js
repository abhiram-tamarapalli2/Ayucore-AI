// ==========================================
// AYUCORE-AI MEDICAL CHATBOT - MAIN JAVASCRIPT
// ==========================================

// ==========================================
// GLOBAL VARIABLES AND CONFIGURATION
// ==========================================

const CONFIG = {
    API_ENDPOINTS: {
        CHAT: '/api/chat',
        RESET: '/reset'
    },
    ANIMATION_DELAYS: {
        TYPING: 100,
        MESSAGE_APPEAR: 300,
        SCROLL: 500
    },
    UI_CONSTANTS: {
        MAX_MESSAGE_LENGTH: 1000,
        SCROLL_THRESHOLD: 100
    }
};

// ==========================================
// UTILITY FUNCTIONS
// ==========================================

// Utility function to safely get element by ID
function getElement(id) {
    const element = document.getElementById(id);
    if (!element) {
        console.warn(`Element with ID '${id}' not found`);
    }
    return element;
}

// Utility function to create element with classes
function createElement(tag, classes = [], textContent = '') {
    const element = document.createElement(tag);
    if (classes.length > 0) {
        element.classList.add(...classes);
    }
    if (textContent) {
        element.textContent = textContent;
    }
    return element;
}

// Utility function to format time
function formatTime(date = new Date()) {
    return date.toLocaleTimeString('en-US', {
        hour: '2-digit',
        minute: '2-digit'
    });
}

// Utility function to validate input
function validateInput(input) {
    if (!input || typeof input !== 'string') {
        return false;
    }
    return input.trim().length > 0 && input.length <= CONFIG.UI_CONSTANTS.MAX_MESSAGE_LENGTH;
}

// ==========================================
// NAVIGATION FUNCTIONALITY
// ==========================================

class NavigationManager {
    constructor() {
        this.navbar = document.querySelector('.navbar');
        this.navToggle = document.querySelector('.nav-toggle');
        this.navMenu = document.querySelector('.nav-menu');
        this.navLinks = document.querySelectorAll('.nav-link');
        
        this.init();
    }

    init() {
        this.setupEventListeners();
        this.setupScrollEffect();
        this.setActiveLink();
    }

    setupEventListeners() {
        // Mobile menu toggle
        if (this.navToggle) {
            this.navToggle.addEventListener('click', () => {
                this.toggleMobileMenu();
            });
        }

        // Close mobile menu when clicking on links
        this.navLinks.forEach(link => {
            link.addEventListener('click', () => {
                this.closeMobileMenu();
            });
        });

        // Close mobile menu when clicking outside
        document.addEventListener('click', (e) => {
            if (!this.navbar.contains(e.target)) {
                this.closeMobileMenu();
            }
        });
    }

    toggleMobileMenu() {
        if (this.navMenu) {
            this.navMenu.classList.toggle('active');
        }
    }

    closeMobileMenu() {
        if (this.navMenu) {
            this.navMenu.classList.remove('active');
        }
    }

    setupScrollEffect() {
        let lastScrollY = window.scrollY;

        window.addEventListener('scroll', () => {
            const currentScrollY = window.scrollY;

            if (this.navbar) {
                if (currentScrollY > 100) {
                    this.navbar.style.background = 'rgba(255, 255, 255, 0.98)';
                    this.navbar.style.boxShadow = '0 2px 20px rgba(0, 0, 0, 0.1)';
                } else {
                    this.navbar.style.background = 'rgba(255, 255, 255, 0.95)';
                    this.navbar.style.boxShadow = 'none';
                }
            }

            lastScrollY = currentScrollY;
        });
    }

    setActiveLink() {
        const currentPath = window.location.pathname;
        this.navLinks.forEach(link => {
            link.classList.remove('active');
            if (link.getAttribute('href') === currentPath) {
                link.classList.add('active');
            }
        });
    }
}

// ==========================================
// SMOOTH SCROLLING AND ANIMATIONS
// ==========================================

class AnimationManager {
    constructor() {
        this.init();
    }

    init() {
        this.setupSmoothScrolling();
        this.setupScrollAnimations();
        this.setupIntersectionObserver();
    }

    setupSmoothScrolling() {
        // Smooth scrolling for anchor links
        document.querySelectorAll('a[href^="#"]').forEach(anchor => {
            anchor.addEventListener('click', (e) => {
                e.preventDefault();
                const target = document.querySelector(anchor.getAttribute('href'));
                if (target) {
                    target.scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        });
    }

    setupScrollAnimations() {
        // Add animation classes to elements
        const animatedElements = document.querySelectorAll(
            '.feature-card, .tech-layer, .timeline-item, .about-feature'
        );

        animatedElements.forEach((element, index) => {
            element.style.opacity = '0';
            element.style.transform = 'translateY(30px)';
            element.style.transition = 'all 0.6s ease';
            element.style.transitionDelay = `${index * 0.1}s`;
        });
    }

    setupIntersectionObserver() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    entry.target.style.opacity = '1';
                    entry.target.style.transform = 'translateY(0)';
                }
            });
        }, {
            threshold: 0.1,
            rootMargin: '0px 0px -50px 0px'
        });

        document.querySelectorAll(
            '.feature-card, .tech-layer, .timeline-item, .about-feature, .coverage-category, .tech-feature-card'
        ).forEach(element => {
            observer.observe(element);
        });
    }

    // Utility method to animate elements
    animateElement(element, animation = 'fadeIn') {
        element.classList.add('animate-' + animation);
        setTimeout(() => {
            element.classList.remove('animate-' + animation);
        }, 600);
    }
}

// ==========================================
// HERO SECTION FUNCTIONALITY
// ==========================================

class HeroManager {
    constructor() {
        this.heroStats = document.querySelectorAll('.stat-number');
        this.medicalCard = document.querySelector('.medical-card');
        this.init();
    }

    init() {
        this.animateStats();
        this.setupMedicalCardDemo();
    }

    animateStats() {
        const observer = new IntersectionObserver((entries) => {
            entries.forEach(entry => {
                if (entry.isIntersecting) {
                    this.countUpAnimation(entry.target);
                }
            });
        });

        this.heroStats.forEach(stat => {
            observer.observe(stat);
        });
    }

    countUpAnimation(element) {
        const target = parseInt(element.textContent.replace(/[^\d]/g, ''));
        const increment = target / 50;
        let current = 0;

        const timer = setInterval(() => {
            current += increment;
            if (current >= target) {
                current = target;
                clearInterval(timer);
            }
            
            // Format the number based on the original content
            if (element.textContent.includes('%')) {
                element.textContent = Math.floor(current) + '%';
            } else if (element.textContent.includes('k')) {
                element.textContent = Math.floor(current) + 'k+';
            } else {
                element.textContent = Math.floor(current);
            }
        }, 20);
    }

    setupMedicalCardDemo() {
        if (!this.medicalCard) return;

        const messages = [
            { type: 'patient', text: 'Hello, I have been having headaches for the past week.' },
            { type: 'doctor', text: 'I understand your concern. Can you describe the headaches? Are they constant or do they come and go?' },
            { type: 'patient', text: 'They usually start in the morning and get worse during the day.' },
            { type: 'doctor', text: 'Based on your symptoms, this could be tension headaches. I recommend staying hydrated, getting adequate sleep, and managing stress. If symptoms persist, please consult a healthcare professional.' }
        ];

        this.animateMedicalDemo(messages);
    }

    animateMedicalDemo(messages) {
        const cardContent = this.medicalCard.querySelector('.card-content');
        if (!cardContent) return;

        let messageIndex = 0;

        const showNextMessage = () => {
            if (messageIndex >= messages.length) {
                setTimeout(() => {
                    cardContent.innerHTML = '';
                    messageIndex = 0;
                    showNextMessage();
                }, 3000);
                return;
            }

            const message = messages[messageIndex];
            const messageElement = createElement('div', ['message', message.type], message.text);
            
            cardContent.appendChild(messageElement);
            messageElement.style.opacity = '0';
            messageElement.style.transform = 'translateY(10px)';
            
            setTimeout(() => {
                messageElement.style.opacity = '1';
                messageElement.style.transform = 'translateY(0)';
            }, 100);

            // Scroll to bottom
            cardContent.scrollTop = cardContent.scrollHeight;

            messageIndex++;
            setTimeout(showNextMessage, 2000);
        };

        // Start the demo after a delay
        setTimeout(showNextMessage, 1000);
    }
}

// ==========================================
// CHATBOT FUNCTIONALITY
// ==========================================

class ChatbotManager {
    constructor() {
        this.chatMessages = getElement('chatMessages');
        this.messageInput = getElement('messageInput');
        this.sendButton = getElement('sendButton');
        this.patientForm = getElement('patientForm');
        this.resetButton = getElement('resetButton');
        this.savePatientInfoBtn = getElement('savePatientInfoBtn');
        
        this.isTyping = false;
        this.messageHistory = [];
        
        this.init();
    }

    init() {
        if (!this.chatMessages || !this.messageInput) {
            return; // Not on chat page
        }

        this.setupEventListeners();
        this.loadWelcomeMessage();
    }

    setupEventListeners() {
        // Send button click
        if (this.sendButton) {
            this.sendButton.addEventListener('click', () => {
                this.sendMessage();
            });
        }

        // Enter key press
        if (this.messageInput) {
            this.messageInput.addEventListener('keypress', (e) => {
                if (e.key === 'Enter' && !e.shiftKey) {
                    e.preventDefault();
                    this.sendMessage();
                }
            });

            // Auto-resize textarea
            this.messageInput.addEventListener('input', () => {
                this.autoResizeTextarea();
            });
        }

        // Reset button
        if (this.resetButton) {
            this.resetButton.addEventListener('click', () => {
                this.resetChat();
            });
        }

        // Save patient info button
        if (this.savePatientInfoBtn) {
            this.savePatientInfoBtn.addEventListener('click', () => {
                this.savePatientInfo();
            });
        }

        // Patient form submission
        if (this.patientForm) {
            this.patientForm.addEventListener('submit', (e) => {
                e.preventDefault();
                this.updatePatientInfo();
            });
        }
    }

    loadWelcomeMessage() {
        const welcomeMessage = {
            text: "Hello! I'm your AI medical assistant. I'm here to help you with health-related questions and provide general medical information. Please note that I cannot replace professional medical advice. How can I assist you today?",
            sender: 'doctor',
            timestamp: new Date()
        };

        this.displayMessage(welcomeMessage);
    }

    async sendMessage() {
        const message = this.messageInput.value.trim();
        
        if (!validateInput(message) || this.isTyping) {
            return;
        }

        // Display user message
        const userMessage = {
            text: message,
            sender: 'patient',
            timestamp: new Date()
        };

        this.displayMessage(userMessage);
        this.messageHistory.push(userMessage);
        
        // Clear input
        this.messageInput.value = '';
        this.autoResizeTextarea();

        // Show typing indicator
        this.showTypingIndicator();

        try {
            // Get patient info
            const patientInfo = this.getPatientInfo();
            
            // Send to backend
            const response = await fetch(CONFIG.API_ENDPOINTS.CHAT, {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    message: message,
                    patient_info: patientInfo
                })
            });

            const data = await response.json();
            
            // Hide typing indicator
            this.hideTypingIndicator();

            if (data.response) {
                const doctorMessage = {
                    text: data.response,
                    sender: 'doctor',
                    timestamp: new Date()
                };

                this.displayMessage(doctorMessage);
                this.messageHistory.push(doctorMessage);
            } else {
                throw new Error('No response received');
            }

        } catch (error) {
            console.error('Error sending message:', error);
            this.hideTypingIndicator();
            
            const errorMessage = {
                text: "I apologize, but I'm experiencing technical difficulties. Please try again in a moment.",
                sender: 'doctor',
                timestamp: new Date(),
                isError: true
            };

            this.displayMessage(errorMessage);
        }
    }

    displayMessage(message) {
        const messageElement = createElement('div', ['message', message.sender]);
        
        // Create message content
        const messageContent = createElement('div', ['message-content']);
        messageContent.textContent = message.text;
        
        // Create timestamp
        const timestamp = createElement('div', ['message-timestamp']);
        timestamp.textContent = formatTime(message.timestamp);
        
        messageElement.appendChild(messageContent);
        messageElement.appendChild(timestamp);

        // Add error styling if needed
        if (message.isError) {
            messageElement.classList.add('error');
        }

        // Add to chat
        this.chatMessages.appendChild(messageElement);
        
        // Animate appearance
        setTimeout(() => {
            messageElement.classList.add('show');
        }, 10);

        // Scroll to bottom
        this.scrollToBottom();
    }

    showTypingIndicator() {
        this.isTyping = true;
        
        const typingElement = createElement('div', ['message', 'doctor', 'typing']);
        typingElement.innerHTML = `
            <div class="typing-indicator">
                <span></span>
                <span></span>
                <span></span>
            </div>
        `;
        
        typingElement.id = 'typingIndicator';
        this.chatMessages.appendChild(typingElement);
        
        setTimeout(() => {
            typingElement.classList.add('show');
        }, 10);
        
        this.scrollToBottom();
    }

    hideTypingIndicator() {
        this.isTyping = false;
        const typingElement = getElement('typingIndicator');
        if (typingElement) {
            typingElement.remove();
        }
    }

    scrollToBottom() {
        setTimeout(() => {
            this.chatMessages.scrollTo({
                top: this.chatMessages.scrollHeight,
                behavior: 'smooth'
            });
        }, CONFIG.ANIMATION_DELAYS.SCROLL);
    }

    autoResizeTextarea() {
        if (this.messageInput) {
            this.messageInput.style.height = 'auto';
            this.messageInput.style.height = Math.min(this.messageInput.scrollHeight, 120) + 'px';
        }
    }

    getPatientInfo() {
        const formData = new FormData(this.patientForm);
        const patientInfo = {};
        
        for (let [key, value] of formData.entries()) {
            if (value.trim()) {
                patientInfo[key] = value.trim();
            }
        }
        
        return patientInfo;
    }

    updatePatientInfo() {
        // This could trigger a notification or update display
        console.log('Patient information updated');
    }

    savePatientInfo() {
        try {
            // Get form data
            const formData = new FormData(this.patientForm);
            const patientInfo = {};
            
            // Convert form data to object
            for (let [key, value] of formData.entries()) {
                if (value.trim()) {
                    patientInfo[key] = value.trim();
                }
            }
            
            // Store in session storage
            localStorage.setItem('patientInfo', JSON.stringify(patientInfo));
            
            // Show success message
            this.showNotification('Patient information saved successfully!', 'success');
            
            // Update the save button to show it's saved
            if (this.savePatientInfoBtn) {
                const originalText = this.savePatientInfoBtn.innerHTML;
                this.savePatientInfoBtn.innerHTML = '<i class="fas fa-check"></i> Saved!';
                this.savePatientInfoBtn.style.background = 'linear-gradient(135deg, #10b981, #059669)';
                
                // Reset button after 2 seconds
                setTimeout(() => {
                    this.savePatientInfoBtn.innerHTML = originalText;
                }, 2000);
            }
            
        } catch (error) {
            console.error('Error saving patient information:', error);
            this.showNotification('Error saving patient information', 'error');
        }
    }

    showNotification(message, type = 'info') {
        // Create notification element
        const notification = createElement('div', ['notification', type]);
        notification.textContent = message;
        
        // Add to page
        document.body.appendChild(notification);
        
        // Show notification
        setTimeout(() => {
            notification.classList.add('show');
        }, 10);
        
        // Remove after 3 seconds
        setTimeout(() => {
            notification.classList.remove('show');
            setTimeout(() => {
                notification.remove();
            }, 300);
        }, 3000);
    }

    async resetChat() {
        try {
            // Clear chat messages
            this.chatMessages.innerHTML = '';
            this.messageHistory = [];
            
            // Reset backend session
            await fetch(CONFIG.API_ENDPOINTS.RESET, {
                method: 'POST'
            });
            
            // Reload welcome message
            this.loadWelcomeMessage();
            
            // Reset patient form
            if (this.patientForm) {
                this.patientForm.reset();
            }
            
        } catch (error) {
            console.error('Error resetting chat:', error);
        }
    }
}

// ==========================================
// FORM VALIDATION AND ENHANCEMENT
// ==========================================

class FormManager {
    constructor() {
        this.forms = document.querySelectorAll('form');
        this.init();
    }

    init() {
        this.setupFormValidation();
        this.setupFormEnhancements();
    }

    setupFormValidation() {
        this.forms.forEach(form => {
            form.addEventListener('submit', (e) => {
                if (!this.validateForm(form)) {
                    e.preventDefault();
                }
            });

            // Real-time validation
            const inputs = form.querySelectorAll('input, select, textarea');
            inputs.forEach(input => {
                input.addEventListener('blur', () => {
                    this.validateInput(input);
                });
            });
        });
    }

    validateForm(form) {
        let isValid = true;
        const inputs = form.querySelectorAll('input[required], select[required], textarea[required]');
        
        inputs.forEach(input => {
            if (!this.validateInput(input)) {
                isValid = false;
            }
        });
        
        return isValid;
    }

    validateInput(input) {
        const value = input.value.trim();
        let isValid = true;
        
        // Remove previous error styling
        input.classList.remove('error');
        
        // Check required fields
        if (input.hasAttribute('required') && !value) {
            isValid = false;
        }
        
        // Check email format
        if (input.type === 'email' && value && !this.isValidEmail(value)) {
            isValid = false;
        }
        
        // Check phone format
        if (input.type === 'tel' && value && !this.isValidPhone(value)) {
            isValid = false;
        }
        
        // Add error styling if invalid
        if (!isValid) {
            input.classList.add('error');
        }
        
        return isValid;
    }

    isValidEmail(email) {
        const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
        return emailRegex.test(email);
    }

    isValidPhone(phone) {
        const phoneRegex = /^[\+]?[1-9][\d]{0,15}$/;
        return phoneRegex.test(phone.replace(/[\s\-\(\)]/g, ''));
    }

    setupFormEnhancements() {
        // Add floating labels effect
        const formGroups = document.querySelectorAll('.form-group');
        formGroups.forEach(group => {
            const input = group.querySelector('input, select, textarea');
            const label = group.querySelector('label');
            
            if (input && label) {
                input.addEventListener('focus', () => {
                    label.classList.add('focused');
                });
                
                input.addEventListener('blur', () => {
                    if (!input.value.trim()) {
                        label.classList.remove('focused');
                    }
                });
                
                // Check if input has value on load
                if (input.value.trim()) {
                    label.classList.add('focused');
                }
            }
        });
    }
}

// ==========================================
// THEME AND ACCESSIBILITY
// ==========================================

class ThemeManager {
    constructor() {
        this.init();
    }

    init() {
        this.setupAccessibilityFeatures();
        this.setupKeyboardNavigation();
    }

    setupAccessibilityFeatures() {
        // Add skip link
        this.addSkipLink();
        
        // Improve focus visibility
        this.improveFocusVisibility();
        
        // Add ARIA labels where needed
        this.addAriaLabels();
    }

    addSkipLink() {
        const skipLink = createElement('a', ['sr-only'], 'Skip to main content');
        skipLink.href = '#main';
        skipLink.style.position = 'absolute';
        skipLink.style.top = '-40px';
        skipLink.style.left = '6px';
        skipLink.style.background = '#000';
        skipLink.style.color = '#fff';
        skipLink.style.padding = '8px';
        skipLink.style.textDecoration = 'none';
        skipLink.style.zIndex = '10000';
        
        skipLink.addEventListener('focus', () => {
            skipLink.style.top = '6px';
        });
        
        skipLink.addEventListener('blur', () => {
            skipLink.style.top = '-40px';
        });
        
        document.body.insertBefore(skipLink, document.body.firstChild);
    }

    improveFocusVisibility() {
        // Add custom focus styles for better visibility
        const style = document.createElement('style');
        style.textContent = `
            .focus-visible:focus {
                outline: 2px solid #3b82f6;
                outline-offset: 2px;
            }
        `;
        document.head.appendChild(style);
    }

    addAriaLabels() {
        // Add ARIA labels to interactive elements that might need them
        const buttons = document.querySelectorAll('button:not([aria-label])');
        buttons.forEach(button => {
            if (button.innerHTML.includes('<i') && !button.textContent.trim()) {
                // Icon-only button, needs aria-label
                if (button.classList.contains('nav-toggle')) {
                    button.setAttribute('aria-label', 'Toggle navigation menu');
                } else if (button.classList.contains('send-btn')) {
                    button.setAttribute('aria-label', 'Send message');
                }
            }
        });
    }

    setupKeyboardNavigation() {
        // Trap focus in mobile menu when open
        const navMenu = document.querySelector('.nav-menu');
        const navToggle = document.querySelector('.nav-toggle');
        
        if (navMenu && navToggle) {
            navToggle.addEventListener('keydown', (e) => {
                if (e.key === 'Escape') {
                    navMenu.classList.remove('active');
                    navToggle.focus();
                }
            });
        }
    }
}

// ==========================================
// PERFORMANCE AND ERROR HANDLING
// ==========================================

class PerformanceManager {
    constructor() {
        this.init();
    }

    init() {
        this.setupErrorHandling();
        this.setupPerformanceOptimizations();
    }

    setupErrorHandling() {
        // Global error handler
        window.addEventListener('error', (e) => {
            console.error('Global error:', e.error);
            this.logError('JavaScript Error', e.error.message, e.filename, e.lineno);
        });

        // Unhandled promise rejection handler
        window.addEventListener('unhandledrejection', (e) => {
            console.error('Unhandled promise rejection:', e.reason);
            this.logError('Promise Rejection', e.reason);
        });
    }

    logError(type, message, filename = '', lineno = '') {
        // In a real application, you would send this to a logging service
        const errorData = {
            type: type,
            message: message,
            filename: filename,
            lineno: lineno,
            timestamp: new Date().toISOString(),
            userAgent: navigator.userAgent,
            url: window.location.href
        };
        
        console.log('Error logged:', errorData);
    }

    setupPerformanceOptimizations() {
        // Lazy load images
        this.setupLazyLoading();
        
        // Debounce scroll events
        this.setupDebouncedScrollEvents();
    }

    setupLazyLoading() {
        const images = document.querySelectorAll('img[data-src]');
        
        if ('IntersectionObserver' in window) {
            const imageObserver = new IntersectionObserver((entries) => {
                entries.forEach(entry => {
                    if (entry.isIntersecting) {
                        const img = entry.target;
                        img.src = img.dataset.src;
                        img.classList.remove('lazy');
                        imageObserver.unobserve(img);
                    }
                });
            });

            images.forEach(img => imageObserver.observe(img));
        } else {
            // Fallback for browsers without IntersectionObserver
            images.forEach(img => {
                img.src = img.dataset.src;
            });
        }
    }

    setupDebouncedScrollEvents() {
        let scrollTimeout;
        
        window.addEventListener('scroll', () => {
            if (scrollTimeout) {
                clearTimeout(scrollTimeout);
            }
            
            scrollTimeout = setTimeout(() => {
                // Scroll-dependent operations
                this.updateScrollProgress();
            }, 16); // ~60fps
        }, { passive: true });
    }

    updateScrollProgress() {
        const scrolled = window.scrollY;
        const maxScroll = document.documentElement.scrollHeight - window.innerHeight;
        const progress = (scrolled / maxScroll) * 100;
        
        // Update progress indicator if it exists
        const progressBar = document.querySelector('.scroll-progress');
        if (progressBar) {
            progressBar.style.width = `${progress}%`;
        }
    }
}

// ==========================================
// INITIALIZATION
// ==========================================

// Initialize all managers when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    try {
        // Initialize core functionality
        new NavigationManager();
        new AnimationManager();
        new ThemeManager();
        new PerformanceManager();
        new FormManager();
        
        // Initialize page-specific functionality
        if (document.querySelector('.hero')) {
            new HeroManager();
        }
        
        if (document.querySelector('.chat-interface')) {
            new ChatbotManager();
        }
        
        console.log('AyuCore-AI website initialized successfully');
        
    } catch (error) {
        console.error('Error initializing website:', error);
    }
});

// ==========================================
// SERVICE WORKER REGISTRATION (PWA Support)
// ==========================================

if ('serviceWorker' in navigator) {
    window.addEventListener('load', () => {
        navigator.serviceWorker.register('/sw.js')
            .then((registration) => {
                console.log('SW registered: ', registration);
            })
            .catch((registrationError) => {
                console.log('SW registration failed: ', registrationError);
            });
    });
}

// ==========================================
// EXPORT FOR MODULE SYSTEMS
// ==========================================

// Export main classes for potential module use
if (typeof module !== 'undefined' && module.exports) {
    module.exports = {
        NavigationManager,
        AnimationManager,
        ChatbotManager,
        HeroManager,
        FormManager,
        ThemeManager,
        PerformanceManager
    };
}
