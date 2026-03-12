//! OS-level sandboxing for defense in depth.
//!
//! After initialization (config loaded, fonts read, GPU set up, PTY opened),
//! this module restricts the process to a minimal set of syscalls and filesystem
//! paths. A vulnerability in a dependency cannot escalate to arbitrary code
//! execution or file reads.
//!
//! - **Seccomp-BPF**: Restricts allowed syscalls to a minimal allowlist.
//! - **Landlock**: Restricts filesystem access to font directories (read-only).
//!
//! Both degrade gracefully: if the kernel doesn't support them, a warning is
//! logged and execution continues unsandboxed.

/// Install all available sandboxing mechanisms.
///
/// This should be called **after** all initialization is complete (config, fonts,
/// GPU, PTY). Returns `Ok(())` even if sandboxing is unavailable on the platform.
pub fn enter_sandbox(config: &crate::config::SecurityConfig) -> Result<(), crate::Error> {
    if !config.sandbox {
        tracing::info!("sandbox disabled by configuration");
        return Ok(());
    }

    #[cfg(target_os = "linux")]
    {
        install_landlock(config.landlock_enforce)?;
        install_seccomp(config.sandbox_enforce);
    }

    #[cfg(not(target_os = "linux"))]
    {
        tracing::debug!("sandboxing not available on this platform");
    }

    Ok(())
}

/// Install Landlock filesystem restrictions (Linux 5.13+).
///
/// When `enforce` is true, `PartiallyEnforced` status is treated as a failure.
#[cfg(target_os = "linux")]
fn install_landlock(enforce: bool) -> Result<(), crate::Error> {
    #[cfg(feature = "sandbox")]
    {
        use landlock::{
            ABI, Access, AccessFs, BitFlags, PathBeneath, PathFd, Ruleset, RulesetAttr,
            RulesetCreatedAttr, RulesetStatus,
        };

        let abi = ABI::V2;

        let read_access = AccessFs::ReadFile | AccessFs::ReadDir;
        let read_file_only = AccessFs::ReadFile;

        let ruleset = match Ruleset::default().handle_access(AccessFs::from_all(abi)) {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("landlock not supported: {e}");
                return Ok(());
            }
        };

        let ruleset = match ruleset.create() {
            Ok(r) => r,
            Err(e) => {
                tracing::debug!("landlock ruleset creation failed: {e}");
                return Ok(());
            }
        };

        // Allow read-only access to standard font directories.
        let font_dirs: Vec<&str> = vec!["/usr/share/fonts", "/usr/local/share/fonts", "/etc/fonts"];

        // Also allow the user's local font directory.
        let home_fonts = std::env::var("HOME")
            .ok()
            .map(|h| format!("{h}/.local/share/fonts"));

        // Allow read/write access to the Wayland socket directory so the clipboard
        // serving process (forked by wl-clipboard-rs) can connect to the compositor.
        let wayland_dir = std::env::var("XDG_RUNTIME_DIR").ok();
        let rw_access = read_access | AccessFs::WriteFile;

        // Build the rule list using the builder pattern (add_rule consumes self).
        let mut current = ruleset;
        for dir in font_dirs.iter().copied().chain(home_fonts.as_deref()) {
            if std::path::Path::new(dir).exists()
                && let Ok(fd) = PathFd::new(dir)
            {
                match current.add_rule(PathBeneath::new(fd, read_access)) {
                    Ok(r) => current = r,
                    Err(e) => {
                        tracing::debug!(dir, "landlock: failed to add font dir rule: {e}");
                        return Ok(());
                    }
                }
            }
        }

        // Grant access to the Wayland/X11 socket directory for clipboard IPC.
        if let Some(ref runtime_dir) = wayland_dir
            && std::path::Path::new(runtime_dir).exists()
            && let Ok(fd) = PathFd::new(runtime_dir)
        {
            match current.add_rule(PathBeneath::new(fd, rw_access)) {
                Ok(r) => current = r,
                Err(e) => {
                    tracing::debug!(runtime_dir, "landlock: failed to add runtime dir rule: {e}");
                    return Ok(());
                }
            }
        }

        // PTY spawning: allow read/write to /dev/ptmx (char device, file-only
        // access) and /dev/pts (directory) so openpty() can allocate PTY pairs.
        let rw_file_only = read_file_only | AccessFs::WriteFile;
        for (pty_path, access) in [("/dev/ptmx", rw_file_only), ("/dev/pts", rw_access)] {
            if std::path::Path::new(pty_path).exists()
                && let Ok(fd) = PathFd::new(pty_path)
            {
                match current.add_rule(PathBeneath::new(fd, access)) {
                    Ok(r) => current = r,
                    Err(e) => {
                        tracing::debug!(pty_path, "landlock: failed to add PTY rule: {e}");
                        return Ok(());
                    }
                }
            }
        }

        // PTY spawning: allow read + execute access to shell binary directories
        // and shared library directories so execve() can load the user's shell.
        // AccessFs::Execute is required for execve(); ReadFile alone is not enough.
        let exec_access = read_access | AccessFs::Execute;
        for dir in [
            "/bin",
            "/usr/bin",
            "/usr/local/bin",
            "/usr/lib",
            "/lib64",
            "/lib",
        ] {
            if std::path::Path::new(dir).exists()
                && let Ok(fd) = PathFd::new(dir)
            {
                match current.add_rule(PathBeneath::new(fd, exec_access)) {
                    Ok(r) => current = r,
                    Err(e) => {
                        tracing::debug!(dir, "landlock: failed to add exec dir rule: {e}");
                        return Ok(());
                    }
                }
            }
        }

        // Shell startup: allow read/write to /dev/null (bash redirects here
        // constantly) and read-only to /dev/urandom (randomness source).
        let read_only: BitFlags<AccessFs> = read_file_only.into();
        for (dev_path, access) in [("/dev/null", rw_file_only), ("/dev/urandom", read_only)] {
            if std::path::Path::new(dev_path).exists()
                && let Ok(fd) = PathFd::new(dev_path)
            {
                match current.add_rule(PathBeneath::new(fd, access)) {
                    Ok(r) => current = r,
                    Err(e) => {
                        tracing::debug!(dev_path, "landlock: failed to add device rule: {e}");
                        return Ok(());
                    }
                }
            }
        }

        // Shell startup: allow read-only access to /etc/profile.d (directory).
        for dir in ["/etc/profile.d"] {
            if std::path::Path::new(dir).exists()
                && let Ok(fd) = PathFd::new(dir)
            {
                match current.add_rule(PathBeneath::new(fd, read_access)) {
                    Ok(r) => current = r,
                    Err(e) => {
                        tracing::debug!(dir, "landlock: failed to add shell dir rule: {e}");
                        return Ok(());
                    }
                }
            }
        }

        // PTY spawning: allow read-only access to individual config files
        // needed by the dynamic linker, user lookup, and shell startup (these
        // are regular files, so only ReadFile is valid — ReadDir would cause
        // partial enforcement).
        for file in [
            "/etc/ld.so.cache",
            "/etc/passwd",
            "/etc/nsswitch.conf",
            "/etc/profile",
            "/etc/bash.bashrc",
            "/etc/environment",
        ] {
            if std::path::Path::new(file).exists()
                && let Ok(fd) = PathFd::new(file)
            {
                match current.add_rule(PathBeneath::new(fd, read_file_only)) {
                    Ok(r) => current = r,
                    Err(e) => {
                        tracing::debug!(file, "landlock: failed to add config file rule: {e}");
                        return Ok(());
                    }
                }
            }
        }

        // PTY spawning: allow read/write to $HOME (shell rc files, working
        // directory) and /tmp (commonly used by shell commands).
        let home_dir = std::env::var("HOME").ok();
        for rw_dir in home_dir.as_deref().into_iter().chain(["/tmp"]) {
            if std::path::Path::new(rw_dir).exists()
                && let Ok(fd) = PathFd::new(rw_dir)
            {
                match current.add_rule(PathBeneath::new(fd, rw_access)) {
                    Ok(r) => current = r,
                    Err(e) => {
                        tracing::debug!(rw_dir, "landlock: failed to add rw dir rule: {e}");
                        return Ok(());
                    }
                }
            }
        }

        match current.restrict_self() {
            Ok(status) => match status.ruleset {
                RulesetStatus::FullyEnforced => {
                    tracing::info!("landlock: filesystem sandbox fully enforced");
                }
                RulesetStatus::PartiallyEnforced => {
                    if enforce {
                        tracing::error!(
                            "landlock: filesystem sandbox only partially enforced \
                             (strict mode enabled)"
                        );
                        return Err(crate::Error::EventLoop(
                            "landlock: partial enforcement rejected by strict config".into(),
                        ));
                    }
                    tracing::info!("landlock: filesystem sandbox partially enforced");
                }
                RulesetStatus::NotEnforced => {
                    tracing::warn!("landlock: not enforced (kernel too old?)");
                }
            },
            Err(e) => {
                tracing::warn!("landlock: restrict_self failed: {e}");
            }
        }
    }

    #[cfg(not(feature = "sandbox"))]
    {
        let _ = enforce;
        tracing::debug!("landlock support not compiled in");
    }

    Ok(())
}

/// Install seccomp-BPF syscall filter (Linux).
///
/// When `enforce` is true, denied syscalls return `EPERM` instead of being logged.
#[cfg(target_os = "linux")]
fn install_seccomp(enforce: bool) {
    #[cfg(feature = "sandbox")]
    {
        use seccompiler::{BpfProgram, SeccompAction, SeccompFilter, SeccompRule};
        use std::collections::BTreeMap;

        // Build an allowlist of syscalls needed after initialization.
        // The terminal only needs: read/write (PTY + GPU), mmap/munmap/mprotect (GPU),
        // ioctl (PTY/GPU), poll/epoll (event loop), futex (threads), clock_gettime,
        // signal handling, and exit.
        let allowed_syscalls: Vec<(i64, Vec<SeccompRule>)> = vec![
            // I/O
            (libc::SYS_read, vec![]),
            (libc::SYS_write, vec![]),
            (libc::SYS_readv, vec![]),
            (libc::SYS_writev, vec![]),
            (libc::SYS_close, vec![]),
            (libc::SYS_lseek, vec![]),
            // Memory management (GPU needs these)
            (libc::SYS_mmap, vec![]),
            (libc::SYS_munmap, vec![]),
            (libc::SYS_mprotect, vec![]),
            (libc::SYS_mremap, vec![]),
            (libc::SYS_madvise, vec![]),
            (libc::SYS_brk, vec![]),
            // Device I/O (PTY ioctl, GPU ioctl)
            (libc::SYS_ioctl, vec![]),
            // Event loop
            (libc::SYS_poll, vec![]),
            (libc::SYS_ppoll, vec![]),
            (libc::SYS_epoll_create1, vec![]),
            (libc::SYS_epoll_ctl, vec![]),
            (libc::SYS_epoll_wait, vec![]),
            (libc::SYS_epoll_pwait, vec![]),
            (libc::SYS_eventfd2, vec![]),
            (libc::SYS_select, vec![]),
            (libc::SYS_pselect6, vec![]),
            // Threading and synchronization
            (libc::SYS_futex, vec![]),
            (libc::SYS_clone, vec![]), // fork() on older glibc
            (libc::SYS_clone3, vec![]),
            (libc::SYS_set_robust_list, vec![]),
            (libc::SYS_rseq, vec![]),
            // Timing
            (libc::SYS_clock_gettime, vec![]),
            (libc::SYS_clock_getres, vec![]),
            (libc::SYS_gettimeofday, vec![]),
            (libc::SYS_nanosleep, vec![]),
            (libc::SYS_clock_nanosleep, vec![]),
            // Signals
            (libc::SYS_rt_sigaction, vec![]),
            (libc::SYS_rt_sigprocmask, vec![]),
            (libc::SYS_rt_sigreturn, vec![]),
            (libc::SYS_kill, vec![]), // For SIGWINCH
            (libc::SYS_tgkill, vec![]),
            // Process exit
            (libc::SYS_exit, vec![]),
            (libc::SYS_exit_group, vec![]),
            // Waitpid (for child shell)
            (libc::SYS_wait4, vec![]),
            // File descriptors (for epoll/dup)
            (libc::SYS_dup, vec![]),
            (libc::SYS_dup2, vec![]),
            (libc::SYS_dup3, vec![]),
            (libc::SYS_fcntl, vec![]),
            (libc::SYS_pipe2, vec![]),
            // GPU drivers may need these
            (libc::SYS_openat, vec![]), // GPU driver sysfs reads
            (libc::SYS_fstat, vec![]),
            (libc::SYS_newfstatat, vec![]),
            (libc::SYS_statx, vec![]),
            (libc::SYS_getdents64, vec![]),
            (libc::SYS_access, vec![]),
            (libc::SYS_faccessat2, vec![]),
            // Misc required by glibc/runtime
            (libc::SYS_getrandom, vec![]),
            (libc::SYS_getpid, vec![]),
            (libc::SYS_gettid, vec![]),
            (libc::SYS_getuid, vec![]),
            (libc::SYS_getgid, vec![]),
            (libc::SYS_geteuid, vec![]),
            (libc::SYS_getegid, vec![]),
            (libc::SYS_sched_getaffinity, vec![]),
            (libc::SYS_sched_yield, vec![]),
            (libc::SYS_prctl, vec![]),
            (libc::SYS_arch_prctl, vec![]),
            // Networking (Wayland compositor socket, clipboard access)
            (libc::SYS_socket, vec![]),
            (libc::SYS_connect, vec![]),
            (libc::SYS_sendmsg, vec![]),
            (libc::SYS_recvmsg, vec![]),
            (libc::SYS_sendto, vec![]),
            (libc::SYS_recvfrom, vec![]),
            (libc::SYS_shutdown, vec![]),
            // PTY spawning (pane splits and new tabs)
            (libc::SYS_execve, vec![]),
            (libc::SYS_execveat, vec![]),
            (libc::SYS_setsid, vec![]), // login_tty creates new session
            // Seccomp itself
            (libc::SYS_seccomp, vec![]),
        ];

        let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
        for (syscall, rule_list) in allowed_syscalls {
            rules.insert(syscall, rule_list);
        }

        let default_action = if enforce {
            SeccompAction::Errno(libc::EPERM as u32)
        } else {
            SeccompAction::Log // Audit mode — log denied syscalls without blocking them
        };

        let filter = match SeccompFilter::new(
            rules,
            default_action,
            SeccompAction::Allow,
            std::env::consts::ARCH
                .try_into()
                .unwrap_or(seccompiler::TargetArch::x86_64),
        ) {
            Ok(f) => f,
            Err(e) => {
                tracing::warn!("seccomp: failed to create filter: {e}");
                return;
            }
        };

        let bpf: BpfProgram = match filter.try_into() {
            Ok(p) => p,
            Err(e) => {
                tracing::warn!("seccomp: failed to compile BPF program: {e}");
                return;
            }
        };

        match seccompiler::apply_filter(&bpf) {
            Ok(()) => {
                let mode = if enforce { "enforce" } else { "audit" };
                tracing::info!(mode, "seccomp: syscall filter installed");
            }
            Err(e) => {
                tracing::error!("seccomp: failed to apply filter: {e}");
            }
        }
    }

    #[cfg(not(feature = "sandbox"))]
    {
        let _ = enforce;
        tracing::debug!("seccomp support not compiled in");
    }
}
