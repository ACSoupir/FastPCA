#!/usr/bin/perl

# Name: monitor_memory.pl
# Purpose: Registers peak DRAM usage of a process
# Usage: perl monitor_memory.pl <PID> <interval_in_seconds> <pidfile>
# Example: perl monitor_memory.pl 12345 1 /tmp/pidfile
# Date: January 19th 2025
# Author: Christos Argyropoulos
# License: MIT https://mit-license.org/

use strict;
use warnings;
use Time::HiRes qw(nanosleep); # high res waiting
use IPC::System::Simple qw(capturex); # safe capture that bypasses the shell

# Process the command line
my ( $pid, $interval_sec, $pidfile ) = @ARGV;
die "Usage: $0 <PID> <interval_in_seconds> <pidfile>\n"
  unless defined $pid && defined $interval_sec && defined $pidfile;

# Write our own PID that R will use to kill the Perl application
open( my $fh, '>', $pidfile ) or die "Can't write to $pidfile: $!";
print $fh "$$\n";
close $fh;

# Obtain initial memory usage
my $interval    = $interval_sec * 1_000_000_000;
my $initial_mem = capturex('ps', '-o', 'rss=', '-p', $pid);
chomp($initial_mem);
my $max_delta = 0;

# Register the INT and TERM signal handlers to print peak and initial DRAM usage
$SIG{INT} = $SIG{TERM} = sub {
    print "$max_delta\t$initial_mem\n";
    exit 0;
};


# Obtain the RSS, store the maximum delta up to this point, sleep and re-awaken
while (1) {
    my $current = capturex('ps', '-o', 'rss=', '-p', $pid);
    chomp($current);
    my $delta = $current - $initial_mem;
    $max_delta = $delta if $delta > $max_delta;
    nanosleep($interval);
}
